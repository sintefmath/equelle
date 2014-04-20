#include <iostream>
#include <string>
#include <vector>

#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"

using namespace equelleCUDA;
typedef Opm::AutoDiffBlock<Scalar> ADB;



// Comparison function:
int compare( CollOfScalar coll, ADB adb, std::string msg) {

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
    std::vector<double> vals = coll.copyToHost();
    for ( int i = 0; i < coll.size(); i++) {
	if ( fabs(vals[i] - v[i]) > 10*std::numeric_limits<double>::epsilon() ) {
	    std::cout << "vals[" << i << "] = " << vals[i];
	    std::cout << "but v[" << i << "] = " << v[i] << "\n";
	    correct = false;
	}
    }
    if (!correct) {
	std::cout << "Error in " << msg << "\n";
	std::cout << "Scalar values are wrong (see above)\n";
	return 1;
    }

    // Comparing matrix:
    if ( coll.useAutoDiff() ) {
	hostMat mat = coll.matrixToHost();
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

	// Vals:
	double* lf_vals = m.valuePtr();
	for (int i = 0; i < mat.vals.size(); ++i) {
	    if ( fabs(mat.vals[i] - lf_vals[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "mat.vals[" << i << "] = " << mat.vals[i];
		std::cout << "but lf_vals[" << i << "] = " << lf_vals[i] << "\n";
		correct = false;
	    }
	}
	if ( !correct ) {
	    std::cout << "Error in matrix in " << msg << "\n";
	    std::cout << "\tThe indices in the val pointer is wrong\n";
	    return 1;
	}
	
	// Row ptr
	int* lf_rowPtr = m.outerIndexPtr();
	for (int i = 0; i < mat.rowPtr.size(); ++i) {
	    if ( fabs(mat.rowPtr[i] - lf_rowPtr[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "mat.rowPtr[" << i << "] = " << mat.rowPtr[i];
		std::cout << "but lf_rowPtr[" << i << "] = " << lf_rowPtr[i] << "\n";
		correct = false;
	    }
	}
	if (!correct) {
	    std::cout << "Error in matrix in " << msg << "\n";
	    std::cout << "\tThe indices in the rowPtr pointer is wrong\n";
	    return 1;
	}

	// Col ind
	int* lf_colInd = m.innerIndexPtr();
	for (int i = 0; i < mat.colInd.size(); ++i) {
	    if ( fabs(mat.colInd[i] - lf_colInd[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "mat.colInd[" << i << "] = " << mat.colInd[i];
		std::cout << "but lf_colInd[" << i << "] = " << lf_colInd[i] << "\n";
		correct = false;
	    }
	}
	if (!correct) {
	    std::cout << "Error in matrix in " << msg << "\n";
	    std::cout << "\tThe indices in the colInd pointer is wrong\n";
	    return 1;
	}
	
    } // Use autodiff
    else {
	std::cout << "Error in " << msg << "\n";
	std::cout << "\tuseAutoDiff() gives false\n";
	return 1;
    }
    
    std::cout << "Test " << msg << " correct\n";

    return 0;
} 



int main(int argc, char** argv) {
    
    if ( argc < 2 ) {
	std::cout << "Need a parameter file\n";
	return 1;
    }

    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);

    // Need the helper ops outside of the runtime as well:
    Opm::GridManager gridMan(14,22,9,1,1,1);
    Opm::HelperOps hops(*(gridMan.c_grid()));
    int numCells = gridMan.c_grid()->number_of_cells;
    std::cout << "Number of cells are: " << numCells << "\n";
    
    // Create an autodiff variable which we want to do tests on:
    ADB::V init_V(numCells);
    std::vector<double> init_vec; 
    for ( int i = 0; i < numCells; ++i) {
	init_V[i] = i;
	init_vec.push_back(i);
    }
    std::vector<int> blocksize = { numCells };
    ADB lf = ADB::variable(0, init_V, blocksize);

    // Init a CollOfScalar:
    CudaArray init_array(init_vec);
    CudaMatrix init_matrix(lf.derivative()[0]);
    CollOfScalar myColl(init_array, init_matrix);

    // Create test for CollOfScalar vs ADB and test 
    if ( compare( myColl, lf, "Init CollOfScalar") ) { return 1; }


    return 0;
}
