#include <iostream>
#include <string>
#include <vector>
#include <array>

#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"
#include "../../serial/include/equelle/EquelleRuntimeCPU.hpp"

#define MY_THROW OPM_THROW(std::runtime_error,"see_above");

using namespace equelleCUDA;
typedef Opm::AutoDiffBlock<Scalar> ADB;

typedef equelle::CollOfScalar SerialCollOfScalar;
typedef equelle::CollOfCell SerialCollOfCell;
//typedef equelle::EquelleRuntimeCPU


int compare( CollOfScalar coll, ADB adb, std::string msg, double tol = 0.0, bool noAD = false);
int matrixCompare( hostMat mat, ADB::M m, std::string msg, double tol = 0.0);

int compareER( CollOfScalar cuda, SerialCollOfScalar serial, std::string msg, double tol = 0.0, bool noAD = false) {
    ADB adb = ADB::function(serial.value(), serial.derivative());
    return compare(cuda, adb, msg, tol, noAD);
}

int compareScalars( double cuda, double serial, std::string msg, double tol = 0.0);

// Comparison function:
int compare( CollOfScalar coll, ADB adb, std::string msg, double tol, bool noAD) {

    if (tol == 0.0) {
	tol = 10;
    }
    tol = tol*std::numeric_limits<double>::epsilon();

    std::cout << "Comparing: " << msg << "\n";
    ADB::V v = adb.value();
    ADB::M m = adb.derivative()[0];
    if ( coll.size() != v.size() ) {
	std::cout << "ERROR in " << msg << "\n";
	std::cout << "coll.size() = " << coll.size() << " while v.size() = ";
	std::cout << v.size() << "\n";
	return 1;
    }
    bool correct = true;
    int errors = 0;
    double diff;
    std::vector<double> vals = coll.copyToHost();
    for ( int i = 0; i < coll.size(); i++) {
	diff = fabs((vals[i] - v[i])/v[i]);
	if ( diff > tol ) {
	    std::cout << "vals[" << i << "] = " << vals[i];
	    std::cout << " but v[" << i << "] = " << v[i];
	    //std::cout << " with diff: " << fabs(vals[i] - v[i]) << "\n";
	    std::cout << " with diff: "<< diff << "\n";
	    correct = false;
	    errors++;
	}
    }
    if (!correct) {
	std::cout << "Error in " << msg << "\n";
	std::cout << "\t" << errors << " scalar values are wrong (see above)\n";
	std::cout << "\tUsed tol = " << tol << "\n";
	return 1;
    }
    
    // Comparing matrix:
    if ( coll.useAutoDiff() ) {
	hostMat mat = coll.matrixToHost();
	if ( matrixCompare( mat, m, msg, tol) ) {
	    return 1;
	}
    } // Use autodiff
    else {
	if ( noAD == false ) {
	    std::cout << "Error in " << msg << "\n";
	    std::cout << "\tuseAutoDiff() gives false\n";
	    return 1;
	}
    }
    
    std::cout << "Test " << msg << " correct\n\n";
    
    return 0;
}

int matrixCompare( hostMat mat, ADB::M m_colMajor, std::string msg, double tol) {
    
    if (tol == 0.0) {
	tol = 10;
    }
    if (tol > 1.0e-7) {
	tol = tol*std::numeric_limits<double>::epsilon();
    }
	
    // ADB::M uses column major format!
    // Cannot compare arrays in column major format with arrays in
    // row major formats!
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> m(m_colMajor);

    if ( mat.nnz != m.nonZeros() ) {
	std::cout << "Error in " << msg;
	std::cout << "Wrong number of nnz: " << mat.nnz;
	std::cout << " should be " << m.nonZeros() << "\n";
	return 1;
    }
    if ( mat.rows != m.rows() ) {
	std::cout << "Error in " << msg;
	std::cout << "Wrong number of rows: " << mat.rows;
	std::cout << " should be " << m.rows() << "\n";
	return 1;
    }
    if ( mat.cols != m.cols() ) {
	std::cout << "Error in " << msg;
	std::cout << "Wrong number of cols: " << mat.cols;
	std::cout << " should be " << m.cols() << "\n";
	return 1;
    }
    
    // Checking values:
    bool correct = true;
    int errors = 0;

    // Vals:
    double* lf_vals = m.valuePtr();
    for (int i = 0; i < mat.vals.size(); ++i) {
	double diff = fabs((mat.vals[i] - lf_vals[i])/lf_vals[i]);
	if ( diff > tol ) {
	     // 100000*std::numeric_limits<double>::epsilon() ) {
	    std::cout << "mat.vals[" << i << "] = " << mat.vals[i];
	    std::cout << " but lf_vals[" << i << "] = " << lf_vals[i];
	    std::cout << " with diff: "<< diff;
	    std::cout << "\n";
	    correct = false;
	    errors++;
	}
    }
    if ( !correct ) {
	std::cout << "Error in matrix in " << msg << "\n";
	std::cout << "\t" << errors << " values in the val pointer is wrong\n";
	std::cout << "\tWith tol = " << tol << "\n";
	return 1;
    }
    
    // Row ptr
    int* lf_rowPtr = m.outerIndexPtr();
    for (int i = 0; i < mat.rowPtr.size(); ++i) {
	if ( fabs(mat.rowPtr[i] - lf_rowPtr[i]) > 10*std::numeric_limits<double>::epsilon() ) {
	    std::cout << "mat.rowPtr[" << i << "] = " << mat.rowPtr[i];
	    std::cout << " but lf_rowPtr[" << i << "] = " << lf_rowPtr[i] << "\n";
	    correct = false;
	    errors++;
	}
    }
    if (!correct) {
	std::cout << "Error in matrix in " << msg << "\n";
	std::cout << "\t" << errors << " indices in the rowPtr pointer is wrong\n";
	return 1;
    }
    
    // Col ind
    int* lf_colInd = m.innerIndexPtr();
    for (int i = 0; i < mat.colInd.size(); ++i) {
	if ( fabs(mat.colInd[i] - lf_colInd[i]) > 10*std::numeric_limits<double>::epsilon() ) {
	    std::cout << "mat.colInd[" << i << "] = " << mat.colInd[i];
	    std::cout << " but lf_colInd[" << i << "] = " << lf_colInd[i] << "\n";
	    correct = false;
	    errors++;
	}
    }
    if (!correct) {
	std::cout << "Error in matrix in " << msg << "\n";
	std::cout << "\t" << errors << " indices in the colInd pointer is wrong\n";
	return 1;
    }
    
    return 0;
} // matrixCompare()


int compareScalars( double cuda, double serial, std::string msg, double tol) {
 
    if (tol == 0.0) {
	tol = 10;
    }
    tol = tol*std::numeric_limits<double>::epsilon();

    std::cout << "\nTesting " << msg << "\n";
    std::cout << "Cuda   : " << cuda << "\n";
    std::cout << "Serial : " << serial << "\n";
    double diff = fabs((cuda - serial)/serial);
    if ( diff > tol ) {
	// 100000*std::numeric_limits<double>::epsilon() ) {
	std::cout << "Differs by "<< diff << " (relative) with tolerance " << tol;
	std::cout << "\n";
	return 1;
    }
    std::cout << "Test " << msg << " correct\n";
    return 0;
}

// Printing function:
void printNonzeros(ADB adb) {
    for (int i = 0; i < adb.derivative()[0].nonZeros(); i++) {
	std::cout << adb.derivative()[0].valuePtr()[i] << "\t";
	if ( i % 8 == 7) {
	    std::cout << "\n";
	}
    }
}

void printNonzeros(SerialCollOfScalar s) {
    printNonzeros( ADB::function(s.value(), s.derivative()));
}


// ------------------------------------------------------------
// -----------------    MAIN    -------------------------------
// ------------------------------------------------------------

int main(int argc, char** argv) {
    
    if ( argc < 2 ) {
	std::cout << "Need a parameter file\n";
	return 1;
    }

    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);
    equelle::EquelleRuntimeCPU serialER(param);

    // Need the helper ops outside of the runtime as well:
    Opm::GridManager gridMan(14,22,9,1,1,1);
    Opm::HelperOps hops(*(gridMan.c_grid()));
    int numCells = gridMan.c_grid()->number_of_cells;
    //int numFaces = gridMan.c_grid()->number_of_faces;
    std::cout << "Number of cells are: " << numCells << "\n";
    
    // Create an autodiff variable which we want to do tests on:
    ADB::V init_V(numCells);
    std::vector<double> init_vec; 
    for ( int i = 0; i < numCells; ++i) {
	init_V[i] = i + (i-(313%17))*0.1;
	//init_vec.push_back(i);
    }
    std::vector<int> blocksize = { numCells };
    ADB initADB = ADB::variable(0, init_V, blocksize);
    
    // Do some wierd stuff to the derivative of myADB, so that
    // it is not just a identity matrix:
    ADB grad_init = hops.grad * initADB;
    ADB myADB = hops.div * grad_init;
    grad_init = hops.grad * myADB;
    myADB = hops.div * grad_init;

    // Create a constant:
    ADB::V const_v(numCells);
    std::vector<double> const_vec;
    for (int i = 0; i < numCells; ++i) {
	const_v[i] = (i%30)*0.1;
	const_vec.push_back(const_v[i]);
	init_vec.push_back(myADB.value()[i]);
    }
    ADB myScalADB = ADB::constant(const_v, blocksize);

    // Init a CollOfScalar:
    CudaArray init_array(init_vec);
    CudaMatrix init_matrix(myADB.derivative()[0]);
    CollOfScalar myColl(init_array, init_matrix);
    CollOfScalar myScal(const_vec);

    // Create test for CollOfScalar vs ADB and test 
    if ( compare( myColl, myADB, "Init CollOfScalar") ) { return 1; }



    //----------- START TESTS --------------- //
    
    // +
    
    // Autodiff + non-autodiff
    CollOfScalar myColl2 = myColl + myScal;
    ADB myADB2 = myADB + myScalADB;
    if (compare(myColl2, myADB2, "adb + non-adb") ) { return 1;}

    // AD + AD
    myColl2 = myColl + myColl2;
    myADB2 = myADB + myADB2;
    if ( compare( myColl2, myADB2, "adb + adb") ) {return 1;}

    // *
    
    // scalar *
    CollOfScalar myColl3 = 3.4 * myColl2;
    ADB myADB3 = 3.4 * myADB2;
    if ( compare( myColl3, myADB3, "3.4 * adb") ) {return 1; }
    
    //  * scalar
    myColl3 = myColl3 * 0.5;
    myADB3 = myADB3 * 0.5;
    if ( compare( myColl3, myADB3, "adb * 0.5") ) { return 1;}
    

    // - 
    // AD - AD
    CollOfScalar myColl4 = myColl2 - myColl3;
    ADB myADB4 = myADB2 - myADB3;
    if ( compare( myColl4, myADB4, "adb - adb") ) { return 1; }

    // AD - AD again
    myColl4 = myColl3 - myColl3;
    myADB4 = myADB3 - myADB3;
    if ( compare( myColl4, myADB4, "adb - adb = zeros") ) {return 1; }

    // AD - nonAD
    myColl4 = myColl3 - myScal;
    myADB4 = myADB3 - myScalADB;
    if ( compare( myColl4, myADB4, "adb - nonADB") ) {return 1; }
    
    // nonAD - AD
    myColl4 = myScal - myColl3;
    myADB4 = myScalADB - myADB3;
    if ( compare( myColl4, myADB4, "nonADB - adb") ) {return 1; }

    // unary minus
    myColl4 = -myColl3;
    myADB4 = -1.0*myADB3;
    if ( compare( myColl4, myADB4, "unary minus") ) {return 1;}
    
    //printNonzeros(myADB4);

    // /
    
    // / scalar
    CollOfScalar myColl5 = myColl4 / 0.25;
    ADB myADB5 = myADB4;
    myADB5 = myADB5 * 4.0;
    if ( compare( myColl5, myADB5, "adb / scalar") ) {return 1; }
    
    //  ------------ * -----------------------
    // Since multiplication is not working at first attempt, we have lots of 
    // tests here... 
    // What was wrong: ADB::M is column major and only at the point
    // of multiplication did we get non-symmetric matrices that made it visible!

    // Check that input is okey
    if ( compare( myColl2, myADB2, "checking nr 2") ) {return 1;}
    if ( compare( myColl5, myADB5, "checking nr 5") ) {return 1;}
    
    // Identity matrix
    ADB::M eye_adb = initADB.derivative()[0];
    CudaMatrix eye_cuda(numCells);
    // Check that we have identity:
    if ( matrixCompare( eye_cuda.toHost(), eye_adb, "Identity matrix") ) {return 1;}
    std::cout << "Identity matrix passed\n";

    // Identity matrix * matrix
    ADB::M eye_myADB5 = eye_adb * myADB5.derivative()[0];
    CudaMatrix eye_myColl5 = eye_cuda * myColl5.derivative();
    if ( matrixCompare( eye_myColl5.toHost(), eye_myADB5, "Identity matrix * matrix")) {return 1;}
    std::cout << "Identity matrix * matrix passed\n";
    
    // Matrix * identity matrix
    ADB::M myADB5_eye = myADB5.derivative()[0] * eye_adb;
    CudaMatrix myColl5_eye = myColl5.derivative() * eye_cuda;
    if ( matrixCompare( myColl5_eye.toHost(), myADB5_eye, "matrix * identity matrix")) {return 1;}
    std::cout << "matrix * identity matrix passed\n";


    // Check matrix multiplication
    ADB::M m_test = myADB2.derivative()[0] * myADB5.derivative()[0];
    CudaMatrix cuda_m_test = myColl2.derivative() * myColl5.derivative();
    if (matrixCompare( cuda_m_test.toHost(), m_test, "Matrix mult test")) { return 1; }
    //                                                                   300000
    std::cout << "Matrix mult test passed\n";

    // Check diagonal matrix * matrix
    typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> D;
    D diag_test = myADB2.value().matrix().asDiagonal();
    ADB::M diagMatrix = diag_test * myADB5.derivative()[0];
    CudaMatrix cuda_diag_test(myColl2);
    CudaMatrix cuda_diagMatrix = cuda_diag_test * myColl5.derivative();
    if ( matrixCompare( cuda_diagMatrix.toHost(), diagMatrix, "diagMatrix * matrix")) {return 1; }
    std::cout << "diagMatrix * matrix passed\n";

    // Check AD * AD
    CollOfScalar myColl6 = myColl2 * myColl5;
    ADB myADB6 = myADB2 * myADB5;
    if ( compare( myColl6, myADB6, "AD * AD") ) {return 1; }

    // Check nonAD * AD;
    CollOfScalar myColl7 = myScal * myColl6;
    ADB myADB7 = myScalADB * myADB6;
    if ( compare( myColl7, myADB7, "nonAD * AD") ) {return 1; }

    // Check AD * nonAD
    myColl7 = myColl7 * myScal;
    myADB7 = myADB7 * myScalADB;
    if ( compare( myColl7, myADB7, "AD * nonAD") ) {return 1; }

    //printNonzeros(myADB7);


    // Division: /
    
    // Check AD / AD:
    CollOfScalar myColl8 = myColl7 / myColl6;
    ADB myADB8 = myADB7 / myADB6;
    if ( compare( myColl8, myADB8, "AD / AD") ) { return 1; }
    
    // Check AD / nonAD
    CollOfScalar myColl9 = myColl7 / myScal;
    ADB myADB9 = myADB7 / myScalADB;
    if ( compare( myColl9, myADB9, "AD / nonAD") ) {return 1; }
    
    // Check nonAD / AD
    myColl9 = myScal / myColl6;
    myADB9 = myScalADB / myADB6;
    if ( compare( myColl9, myADB9, "nonAD / AD") ) { return 1; }

    // Check scalar / AD
    // Can't test this as "scalar / ADB" is not implemented...
    CollOfScalar myColl10 = 1000000 / myColl6;
    SerialCollOfScalar serial_myColl6(myADB6);
    SerialCollOfScalar serial_myColl10 = 1000000 / serial_myColl6;
    ADB myADB10 = ADB::function(serial_myColl10.value(), serial_myColl10.derivative());
    if ( compareER( myColl10, serial_myColl10, "scalar / AD") ) {return 1; }


    // On 
    CollOfScalar myOn_cuda = er.operatorOn( myColl10, er.allCells(), er.interiorCells());
    SerialCollOfScalar myOn_serial = serialER.operatorOn( serial_myColl10, 
							  serialER.allCells(),
							  serialER.interiorCells());
    if ( compareER( myOn_cuda, myOn_serial, "myColl10 On InteriorCells()") ) { return 1; }
    
    // Extend
    CollOfScalar myExt_cuda = er.operatorExtend( myOn_cuda, er.interiorCells(),
						 er.allCells() );
    SerialCollOfScalar myExt_serial = serialER.operatorExtend( myOn_serial,
							       serialER.interiorCells(),
							       serialER.allCells() );
    if ( compareER( myExt_cuda, myExt_serial, "myOn On Extend") ) { return 1; }


    // GRID OPERATIONS

    // Gradient:
    CollOfScalar myGrad_cuda = er.gradient(myColl3);
    SerialCollOfScalar myGrad_serial = serialER.gradient(SerialCollOfScalar(myADB3));
    if ( compareER( myGrad_cuda, myGrad_serial, "Gradient(myColl3)") ) { return 1; }
    // myColl9 creates difficulties...

    // Divergence:
    std::cout << "\nmyGrad_cuda.useAutoDiff() = " << myGrad_cuda.useAutoDiff() << "\n";
    CollOfScalar myDiv_cuda = er.divergence(myGrad_cuda);
    SerialCollOfScalar myDiv_serial = serialER.divergence(myGrad_serial);
    if ( compareER( myDiv_cuda, myDiv_serial, "Divergence(myGrad)") ) { return 1; }

    // Full divergence:
    // Put 3.14 on the boundary
    // BUT WE NEED OPERATOR EXTEND FIRST!
        CollOfScalar cuda_edge = er.operatorExtend(er.operatorExtend(3.14, er.boundaryFaces()), er.boundaryFaces(), er.allFaces()) + (er.operatorExtend(myGrad_cuda, er.interiorFaces(), er.allFaces()));
    CollOfScalar cuda_fulldiv = er.divergence(cuda_edge);
   
    SerialCollOfScalar serial_edge = serialER.operatorExtend(serialER.operatorExtend(3.14, serialER.boundaryFaces()), serialER.boundaryFaces(), serialER.allFaces()) + (serialER.operatorExtend(myGrad_serial, serialER.interiorFaces(), serialER.allFaces()));
    SerialCollOfScalar serial_fulldiv = serialER.divergence(serial_edge);
    
    if ( compareER(cuda_fulldiv, serial_fulldiv, "Divergence(AllFaces())",100) ) { return 1; }

    // Want to test an evaluate On operation
    // x_fulldiv is defined on all cells. 
    // We will a variable defined on boundaryFaces holding the values of inner-cells.
    CollOfCell cuda_inner_cells = er.trinaryIf( er.isEmpty(er.firstCell(er.boundaryFaces())), er.secondCell(er.boundaryFaces()), er.firstCell(er.boundaryFaces()) );
    SerialCollOfCell serial_inner_cells = serialER.trinaryIf( serialER.isEmpty(serialER.firstCell(serialER.boundaryFaces())), serialER.secondCell(serialER.boundaryFaces()), serialER.firstCell(serialER.boundaryFaces()) );
   
    CollOfScalar cuda_inner_cells_vals = er.operatorOn( cuda_fulldiv, er.allCells(), cuda_inner_cells);
    SerialCollOfScalar serial_inner_cells_vals = serialER.operatorOn( serial_fulldiv, serialER.allCells(), serial_inner_cells);
    if ( compareER(cuda_inner_cells_vals, serial_inner_cells_vals, "Inner Cells Vals", 100) ) {return 1; }
    
    // Subset to subset On operator with overlap
    CollOfScalar cuda_bnd_vals = er.operatorOn( cuda_fulldiv, er.allCells(), er.boundaryCells());
    SerialCollOfScalar serial_bnd_vals = serialER.operatorOn( serial_fulldiv, serialER.allCells(), serialER.boundaryCells());
    CollOfScalar cuda_sub2sub = er.operatorOn( cuda_bnd_vals, er.boundaryCells(), cuda_inner_cells);
    SerialCollOfScalar serial_sub2sub = serialER.operatorOn( serial_bnd_vals, serialER.boundaryCells(), serial_inner_cells);
    if ( compareER(cuda_sub2sub, serial_sub2sub, "Subset On subset", 100)) {return 1; }
 
    // SQRT
    CollOfScalar myColl4_squared = myColl4 * myColl4;
    CollOfScalar myColl11 = er.sqrt(myColl4_squared);
    SerialCollOfScalar serial4_squared(myADB4*myADB4);
    SerialCollOfScalar serial11 = serialER.sqrt(serial4_squared);
    if ( compareER( myColl11, serial11, "Sqrt(myColl4*myColl4)") ) { return 1; }


    // TRINARY IF
    double midValue = er.minReduce(myColl11) + 2.0;
    CollOfScalar myColl12 = er.trinaryIf( (myColl11 > midValue), myColl11, er.operatorExtend(double(1000), er.allCells()));
    SerialCollOfScalar serial12 = serialER.trinaryIf( (serial11 > midValue), serial11, serialER.operatorExtend(double(1000), serialER.allCells()));
    if ( compareER( myColl12, serial12, "myColl12 = myColl11 > midValue ? myColl11 : 1000 Extend AllCells()")) { return 1; }
    std::cout << "Before trinary if max " << er.maxReduce(myColl11) << " and min " << er.minReduce(myColl11) << "\n";
    std::cout << "After trinary if max " << er.maxReduce(myColl12) << " and min " << er.minReduce(myColl12) << "\n";

    // myGrad_cuda, myGrad_serial is On InteriorFaces()
    // Trinary If on interiorFaces
    midValue = (er.minReduce(myGrad_cuda) + er.maxReduce(myGrad_cuda))/2.0;
    CollOfScalar myColl13_intf = er.trinaryIf( (myGrad_cuda > 0.05), myGrad_cuda, er.operatorExtend(double(1000), er.interiorFaces()));
    SerialCollOfScalar serial13_intf = serialER.trinaryIf( (myGrad_serial > 0.05), myGrad_serial, serialER.operatorExtend( double(1000), serialER.interiorFaces()));
    if ( compareER( myColl13_intf, serial13_intf, "trinaryIf on InteriorFaces()")) { return 1;}
	std::cout << "Before trinary if max " << er.maxReduce(myGrad_cuda) << " and min " << er.minReduce(myGrad_cuda) << "\n";
    std::cout << "After trinary if max " << er.maxReduce(myColl13_intf) << " and min " << er.minReduce(myColl13_intf) << "\n";


    //printNonzeros(serial_fulldiv);
    CollOfScalar myTri_cuda = er.trinaryIf( cuda_fulldiv > 0, 
					    2.4 * cuda_fulldiv,
					    -1.2 * cuda_fulldiv );
    SerialCollOfScalar myTri_serial = serialER.trinaryIf ( serial_fulldiv > 0,
							   2.4 * serial_fulldiv,
							   -1.2 * serial_fulldiv);
    if ( compareER( myTri_cuda, myTri_serial, "TrinaryIf", 100) ) { return 1; }


    

    // REDUCTIONS
    double cuda_sum = er.sumReduce(myTri_cuda);
    double serial_sum = serialER.sumReduce(myTri_serial);
    if ( compareScalars(cuda_sum, serial_sum, "SumReduce(myTri)") ) { return 1;}
    
    double cuda_min = er.minReduce(myTri_cuda);
    double serial_min = serialER.minReduce(myTri_serial);
    if ( compareScalars(cuda_min, serial_min, "MinReduce(myTri)") ) { return 1; }
    
    double cuda_max = er.maxReduce(myTri_cuda);
    double serial_max = serialER.maxReduce(myTri_serial);
    if ( compareScalars( cuda_max, serial_max, "MaxReduce(myTri)") ) { return 1; }
    
    double cuda_prod = er.prodReduce(0.001*er.operatorOn(myTri_cuda, er.allCells(), er.boundaryCells()));
    double serial_prod = serialER.prodReduce(0.001*serialER.operatorOn(myTri_serial, serialER.allCells(), serialER.boundaryCells()));
    if ( compareScalars( cuda_prod, serial_prod, "ProdReduce(myTri)",100) ) { return 1; }
    
    double cuda_norm = er.twoNormTester(myColl4);
    ADB::V norm_vector = myADB4.value();
    double serial_norm = 0;
    for ( int i = 0; i < norm_vector.size(); i++) {
	serial_norm += norm_vector[i]*norm_vector[i];
    }
    serial_norm = std::sqrt(serial_norm);
    if (compareScalars( cuda_norm, serial_norm, "twoNorm(myColl4)", 13) ) { return 1; }



    // Lambda function test, with arrays of CollOfScalar
    // 1) create set intCells()
    // 2) create function (array CollOfScalar , array serialCollOfScalar) -> array CollOfScalar
    // 2.1) Test input
    // 2.2) Test access to intCells in function
    // 2.3) Test steps in function
    // 3) Create arrays
    // 4) Call function
    // 5) Test output against serial computed functionality outside of lambda.

    // 1)
    CollOfCell intCell = er.interiorCells();
    // 2)
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&, const std::array<SerialCollOfScalar, 3>&)> test_function = [&](const std::array<CollOfScalar, 3>& cudaArrayIn, const std::array<SerialCollOfScalar,3>& serialArrayIn) -> std::array<CollOfScalar, 3> {
	bool ad = !cudaArrayIn[0].useAutoDiff();
	
	// 2.1)
	if (compareER(cudaArrayIn[0], serialArrayIn[0], "cudaArrayIn[0]",0, ad)) {MY_THROW}
	if (compareER(cudaArrayIn[1], serialArrayIn[1], "cudaArrayIn[1]",0, ad)) {MY_THROW}
	if (compareER(cudaArrayIn[2], serialArrayIn[2], "cudaArrayIn[2]",0, ad)) {MY_THROW}

	// 2.2)
	CollOfScalar input1_intc = er.operatorOn(cudaArrayIn[0], er.allCells(), intCell);
	SerialCollOfScalar sinput1_intc = serialER.operatorOn(serialArrayIn[0], serialER.allCells(), serialER.interiorCells());
	if ( compareER(input1_intc, sinput1_intc, "On interiorCells in lambda",0, ad)){MY_THROW}
	double unchanged_midVal = (er.minReduce(input1_intc) + er.maxReduce(input1_intc))/2;
	CollOfScalar input2_intc = er.trinaryIf( (input1_intc > unchanged_midVal), input1_intc, er.operatorExtend(double(1000), intCell));
	SerialCollOfScalar sinput2_intc = serialER.trinaryIf( (sinput1_intc > unchanged_midVal), sinput1_intc, serialER.operatorExtend(double(1000), serialER.interiorCells()));
	if ( compareER(input2_intc, sinput2_intc, "trinary if in lambda",0, ad)) {MY_THROW}
	
	CollOfScalar to_output = er.operatorExtend(input2_intc, er.interiorCells(), er.allCells());
	SerialCollOfScalar serial_to_output = serialER.operatorExtend(sinput2_intc, serialER.interiorCells(), serialER.allCells());
	if ( compareER(to_output, serial_to_output, "Extend in lambda function",0, ad)){MY_THROW}

	return makeArray(cudaArrayIn[0], to_output, cudaArrayIn[2]);
    };
    
    // 3)
    std::array<CollOfScalar, 3> cudaArray = makeArray(myColl11, myColl6, myColl7);
    //if ( CollOfScalar(myColl11.value()).useAutoDiff() ) { MY_THROW}
    std::array<SerialCollOfScalar, 3> serialArray = equelle::makeArray(serial11, SerialCollOfScalar(myADB6), SerialCollOfScalar(myADB7));
    if (compareER(cudaArray[0], serialArray[0], "cudaArray[0]")) {MY_THROW}
    if (compareER(cudaArray[1], serialArray[1], "cudaArray[1]")) {MY_THROW}
    if (compareER(cudaArray[2], serialArray[2], "cudaArray[2]")) {MY_THROW}
    // 4)
    std::array<CollOfScalar, 3> myOutput = test_function(cudaArray, serialArray);
    // 5)
    if (compareER(myOutput[0], serialArray[0], "myOutput[0]")) {MY_THROW}
    // this one is changed
    // if (compareER(myOutput[1], serialArray[1], "myOutput[1]")) {MY_THROW}
    if (compareER(myOutput[2], serialArray[2], "myOutput[2]")) {MY_THROW}
    // Recomputation of myOutput[1] serially:
    SerialCollOfScalar recomp_intc = serialER.operatorOn(serialArray[0], serialER.allCells(), serialER.interiorCells());
    double unchanged_midVal = (serialER.minReduce(recomp_intc) + serialER.maxReduce(recomp_intc))/2;
    SerialCollOfScalar recomp2_intc = serialER.trinaryIf( (recomp_intc > unchanged_midVal), recomp_intc, serialER.operatorExtend(double(1000), serialER.interiorCells()));
    SerialCollOfScalar recomp_fin = serialER.operatorExtend(recomp2_intc, serialER.interiorCells(), serialER.allCells());
    if (compareER(myOutput[1], recomp_fin, "myOutput[1]")) {MY_THROW}

    
    // Call test_function with non-autodiff CollOfScalars
    // 3)
    std::array<CollOfScalar, 3> cudaArrayS = makeArray(CollOfScalar(myColl11.value()), CollOfScalar(myColl6.value()), CollOfScalar(myColl7.value()));
    if (compareER(cudaArrayS[0], serialArray[0], "cudaArray[0] noAD",0, true)) {MY_THROW}
    if (compareER(cudaArrayS[1], serialArray[1], "cudaArray[1] noAD",0, true)) {MY_THROW}
    if (compareER(cudaArrayS[2], serialArray[2], "cudaArray[2] noAD",0, true)) {MY_THROW}
    // 4)
    std::cout << "--------- TEST_FUNCTION without AD ---------\n";
    std::array<CollOfScalar, 3> myOutputS = test_function(cudaArray, serialArray);
    std::cout << "--------- TEST_FUNCTION without AD done ----\n";
    // 5)
    if (compareER(myOutputS[0], serialArray[0], "myOutput[0]")) {MY_THROW}
    // this one is changed
    // if (compareER(myOutput[1], serialArray[1], "myOutput[1]")) {MY_THROW}
    if (compareER(myOutputS[2], serialArray[2], "myOutput[2] noAD")) {MY_THROW}
    if (compareER(myOutputS[1], recomp_fin, "myOutput[1] noAD",0, true)) {MY_THROW}
    

    // Test what fails in the shallow water simulator:
    // q = h + b
    // raw = q - b # = h
    // water1 = (raw > 0.05) ? raw : 1000 Extend AllCells() # fails
    // water2 = (h > 0.05) ? h : 1000 Extend AllCells() # correct.


    std::cout << "\nEnd of main\n";
    return 0;
}

