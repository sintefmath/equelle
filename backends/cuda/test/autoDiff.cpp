#include <iostream>
#include <string>
#include <vector>

#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"
#include "../../serial/include/equelle/EquelleRuntimeCPU.hpp"

using namespace equelleCUDA;
typedef Opm::AutoDiffBlock<Scalar> ADB;

typedef equelle::CollOfScalar SerialCollOfScalar;
//typedef equelle::EquelleRuntimeCPU


int compare( CollOfScalar coll, ADB adb, std::string msg, double tol = 0.0);
int matrixCompare( hostMat mat, ADB::M m, std::string msg, double tol = 0.0);

int compareER( CollOfScalar cuda, SerialCollOfScalar serial, std::string msg, double tol = 0.0) {
    ADB adb = ADB::function(serial.value(), serial.derivative());
    return compare(cuda, adb, msg, tol);
}



// Comparison function:
int compare( CollOfScalar coll, ADB adb, std::string msg, double tol) {

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
	//return 1;
    }
    
    // Comparing matrix:
    if ( coll.useAutoDiff() ) {
	hostMat mat = coll.matrixToHost();
	if ( matrixCompare( mat, m, msg, tol) ) {
	    return 1;
	}
    } // Use autodiff
    else {
	std::cout << "Error in " << msg << "\n";
	std::cout << "\tuseAutoDiff() gives false\n";
	return 1;
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
    CollOfScalar myGrad_cuda = er.gradient(myColl9);
    ADB myGrad_adb = hops.grad * myADB9;
    if ( compare( myGrad_cuda, myGrad_adb, "Gradient(myColl9)") ) { return 1; }

    // Divergence:
    std::cout << "\nmyGrad_cuda.useAutoDiff() = " << myGrad_cuda.useAutoDiff() << "\n";
    CollOfScalar myDiv_cuda = er.divergence(myGrad_cuda);
    ADB myDiv_adb = hops.div * myGrad_adb;
    if ( compare( myDiv_cuda, myDiv_adb, "Divergence(myGrad)", 1000) ) { return 1; }

    // Full divergence:
    // Put 3.14 on the boundary
    // BUT WE NEED OPERATOR EXTEND FIRST!
    /*    CollOfScalar cuda_edge = er.operatorExtend(er.operatorExtend(3.14, er.boundaryFaces()), er.boundaryFaces(), er.allFaces()) + (er.operatorExtend(myGrad_cuda, er.interiorFaces(), er.allFaces()));
    CollOfScalar cuda_fulldiv = er.divergence(cuda_edge);
   
    SerialCollOfScalar serial_edge = serialER.operatorExtend(serialER.operatorExtend(3.14, serialER.boundaryFaces()), serialER.boundaryFaces(), serialER.allFaces()) + (serialER.operatorExtend(SerialCollOfScalar(myGrad_adb), serialER.interiorFaces(), serialER.allFaces()));
    SerialCollOfScalar serial_fulldiv = serialER.divergence(serial_edge);
    
    if ( compareER(cuda_fulldiv, serial_fulldiv, "Divergence(AllFaces())", 100) ) { return 1; }*/

    
    // SQRT
    CollOfScalar myColl4_squared = myColl4 * myColl4;
    CollOfScalar myColl11 = er.sqrt(myColl4_squared);
    SerialCollOfScalar serial4_squared(myADB4*myADB4);
    SerialCollOfScalar serial11 = serialER.sqrt(serial4_squared);
    if ( compareER( myColl11, serial11, "Sqrt(myColl4*myColl4)") ) { return 1; }

    return 0;
}
