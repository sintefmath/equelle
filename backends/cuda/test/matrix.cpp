#include <opm/core/grid/GridManager.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

#include <iostream>
#include <vector>
#include <string>

#include "EquelleRuntimeCUDA.hpp"
#include "CudaMatrix.hpp"

using namespace equelleCUDA;
using std::endl;
using std::cout;
using std::vector;
using std::string;
/* 
   This test file is ment to test the class CudaMatrix!

   It relays on the cusparse library which needs a global 
   variable created by the EquelleRuntimeCUDA constructor.
   Therefore, we need to read the grid.
   
*/




int matrixCompare(hostMat mat, hostMat lf, string msg) {
    if ( mat.nnz != lf.nnz) {
	cout << "Error in matrix.cpp - testing " <<  msg << endl; 
	cout << "mat.nnz = " << mat.nnz << " but should be " << lf.nnz;
	cout << endl;
	return 1;
    }
    if ( mat.rows != lf.rows ) {
	cout << "Error in matrix.cpp - testing " << msg << endl;
	cout << "mat.rows = " << mat.rows << " but should be " << lf.rows;
	cout << endl;
	return 1;
    }
    if ( mat.cols != lf.cols ) {
	cout << "Error in matrix.cpp - testing " << msg << endl;
	cout << "mat.cols = " << mat.cols << " but should be " << lf.cols;
	cout << endl;
	return 1;
    }

    bool correct = true;

    // Vals:
    for (int i = 0; i < mat.vals.size(); ++i) {
	std::cout << mat.vals[i] << " ";
	if (i < lf.vals.size()) {
	    if ( fabs(mat.vals[i] - lf.vals[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "(<- " << lf.vals[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in matrix.cpp - testing " << msg << "\n";
	std::cout << "\tThe indices in the val pointer is wrong\n";
	return 1;
    }

    // Row ptr
    for (int i = 0; i < mat.rowPtr.size(); ++i) {
	std::cout << mat.rowPtr[i] << " ";
	if (i < lf.rowPtr.size()) {
	    if ( fabs(mat.rowPtr[i] - lf.rowPtr[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "(<- " << lf.rowPtr[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in matrix.cpp - testing " << msg << "\n";
	std::cout << "\tThe indices in the rowPtr pointer is wrong\n";
	return 1;
    }

    // Col ind
    for (int i = 0; i < mat.colInd.size(); ++i) {
	std::cout << mat.colInd[i] << " ";
	if (i < lf.colInd.size()) {
	    if ( fabs(mat.colInd[i] - lf.colInd[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "(<- " << lf.colInd[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in matrix.cpp - testing " << msg << "\n";
	std::cout << "\tThe indices in the colInd pointer is wrong\n";
	return 1;
    }

    // Test passed!
    return 0;
}


int main(int argc, char** argv) {

    if (argc < 2) {
	std::cout << "Need a parameter file, please!\n";
	return 1;
    }

    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);

    std::cout << "Creating an empty matrix:\n";
    CudaMatrix A;
    std::cout << "Success! :) \n";
    
    // Create a matrix:
    vector<double> b_v = {3, 4, 1, 4.5, -1, 1, 1};
    vector<int> b_rp = {0, 2, 3, 5, 7};
    vector<int> b_ci = {1,2,3,0,3,2,3};
    hostMat b_lf = {b_v, b_rp, b_ci, 7, 4, 4};
    CudaMatrix B(&b_v[0], &b_rp[0], &b_ci[0], 7, 4, 4);
    if ( matrixCompare(B.toHost(), b_lf, "Init from host and copy to host") ) {
	return 1;
    }

    // Copy constructor:
    CudaMatrix C(B);
    if ( matrixCompare(C.toHost(), b_lf, "Copy constructor") ) {
	return 1;
    }

    // Create another matrix:
    vector<double> c_v = {1,-4,3,4,5,1,-5};
    vector<int> c_rp = {0,2,2,5,7};
    vector<int> c_ci = {0,2,1,2,3,1,3};
    hostMat c_lf = {c_v, c_rp, c_ci, 7,4,4};
    CudaMatrix C_temp(&c_v[0], &c_rp[0], &c_ci[0], 7,4,4);
    if ( matrixCompare(C_temp.toHost(), c_lf, "Init from host and copy to host - again") ) {
	return 1;
    }

    // Copy assignment constructor:
    C = C_temp;
    if ( matrixCompare(C.toHost(), c_lf, "Copy assignment constructor") ) {
	return 1;
    }

    
    CudaMatrix D = B + C;
    vector<double> d_v = {1,3,0,1,4.5,3,4,4,1,1,-4};
    vector<int> d_rp = {0,3,4,8,11};
    vector<int> d_ci = {0,1,2,3,0,1,2,3,1,2,3};
    hostMat d_lf = {d_v, d_rp, d_ci, 11, 4, 4};
    if ( matrixCompare(D.toHost(), d_lf, "B + C") ) {
	return 1;
    }


    return 0;
}


