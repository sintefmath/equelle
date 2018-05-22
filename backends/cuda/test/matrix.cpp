#include <opm/grid/GridManager.hpp>
#include <opm/common/utility/parameters/ParameterGroup.hpp>

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
    cout << "Testing " << msg << "\n";
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
	if ( i < 100 )
	    std::cout << mat.vals[i] << " ";
	if (i == 100 )
	    std::cout << "...\n";
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
	if ( i < 100 )
	    std::cout << mat.rowPtr[i] << " ";
	if (i == 100 )
	    std::cout << "...\n";
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
	if (i < 100)
	    std::cout << mat.colInd[i] << " ";
	if ( i == 100 ) 
	    std::cout << "...\n";
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


// COMPARING ARRAYS

int arrayCompare( const std::vector<double>& host, 
		  const std::vector<double>&lf, 
		  const std::string& msg ) {
    
    cout << "Testing " << msg << "\n";

    if ( host.size() != lf.size()) {
	cout << "Error in matrix.cpp - testing " <<  msg << endl; 
	cout << "host.size() = " << host.size() << " but should be " << lf.size();
	cout << endl;
	return 1;
    }
    bool correct = true;

    // Vals:
    for (int i = 0; i < host.size(); ++i) {
	if ( i < 100 )
	    std::cout << host[i] << " ";
	if (i == 100 )
	    std::cout << "...\n";
	if (i < lf.size()) {
	    if ( fabs(host[i] - lf[i]) > 10*std::numeric_limits<double>::epsilon() ) {
		std::cout << "(<- " << lf[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in matrix.cpp - testing " << msg << "\n";
	std::cout << "\tThe indices in the array is wrong\n";
	return 1;
    }
    
    // Test passed:
    return 0;
}




// ------------   START  MAIN    ------------------


int main(int argc, char** argv) {

    if (argc < 2) {
	std::cout << "Need a parameter file, please!\n";
	return 1;
    }

    Opm::ParameterGroup param( argc, argv, false);
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

    // +
    CudaMatrix D = B + C;
    vector<double> d_v = {1,3,0,1,4.5,3,4,4,1,1,-4};
    vector<int> d_rp = {0,3,4,8,11};
    vector<int> d_ci = {0,1,2,3,0,1,2,3,1,2,3};
    hostMat d_lf = {d_v, d_rp, d_ci, 11, 4, 4};
    if ( matrixCompare(D.toHost(), d_lf, "D = B + C") ) {
	return 1;
    }

    // -
    CudaMatrix F = B - C;
    vector<double> f_v = {-1,3,8,1,4.5,-3,-4,-6, -1,1,6};
    hostMat f_lf = {f_v, d_rp, d_ci, 11, 4, 4};
    if ( matrixCompare(F.toHost(), f_lf, "F = B - C") ) {
	return 1;
    }

    // Sparse matrix multiplication
    CudaMatrix E = B*C;
    vector<double> e_v = {12,16,20,1,-5,4.5,-1,-18,5,4,4,0};
    vector<int> e_rp = {0,3,5,9,12};
    vector<int> e_ci = {1,2,3,1,3,0,1,2,3,1,2,3};
    hostMat e_lf = {e_v, e_rp, e_ci, 12, 4, 4};
    if ( matrixCompare(E.toHost(), e_lf, "E = B * C") ) {
	return 1;
    }

    // Scalar * matrix
    CudaMatrix G = -4*E;
    vector<double> g_v = {-48,-64,-80,-4,20,-18,4,72,-20,-16,-16,0};
    hostMat g_lf = {g_v, e_rp, e_ci, 12, 4, 4};
    if ( matrixCompare(G.toHost(), g_lf, "G = -4*E" ) ) {
	return 1;
    }
    
    // Matrix * Scalar
    CudaMatrix G2 = E*(-4);
    if ( matrixCompare(G2.toHost(), g_lf, "G2 = E*(-4)") ) {
	return 1;
    }

   
    // Create matrix of size 3*5;
    vector<double> wrong_v = {3, 4, 1, 4.5, -1, 1, 1};
    vector<int> wrong_rp = {0, 2, 5, 7};
    vector<int> wrong_ci = {1,2,2,3,4,0,3};
    //hostMat wrong_lf = {wrong_v, wrong_rp, wrong_ci, 7, 4, 4};
    CudaMatrix wrong(&wrong_v[0], &wrong_rp[0], &wrong_ci[0], 7, 3, 5);

    CudaMatrix H;
    bool correctCode = false;
    try {
	H = B + wrong;
    } catch (const std::runtime_error exc) {
	cout << "Threw exception on H = B + wrong\n" << exc.what() << "\n";
	cout << "Good code!\n\n";
	correctCode = true;
    }
    if ( !correctCode ) {
	cout << "Did not throw exception on H = B + wrong\n";
	return 1;
    }

    correctCode = false;
    try {
	H = B * wrong;
    } catch (const std::runtime_error exc) {
	cout << "Threw exception on H = B * wrong\n" << exc.what() << "\n";
	cout << "Good code!\n\n";
	correctCode = true;
    }
    if ( !correctCode ) {
	cout << "Did not throw exception on H = B * wrong\n";
	return 1;
    }

    // Create an identity matrix
    // Create matrix of size 4*4
    vector<double> i_v = {1,1,1,1};
    vector<int> i_rp = {0, 1, 2, 3, 4};
    vector<int> i_ci = {0, 1, 2, 3};
    hostMat i_lf = {i_v, i_rp, i_ci, 4,4,4};
    CudaMatrix I(4);
    if ( matrixCompare(I.toHost(), i_lf, "Identity matrix I(4)") ) {
	return 1;
    }

    CudaMatrix J = -G;
    vector<double> j_v = {48,64,80,4,-20,18,-4,-72,20,16,16,0};
    hostMat j_lf = {j_v, e_rp, e_ci, 12, 4, 4};
    if ( matrixCompare(J.toHost(), j_lf, "J = -G" ) ) {
	return 1;
    }

    // Create an empty matrix and do tests with it
    CudaMatrix empty;
    if ( !empty.isEmpty() ) {
	std::cout << "Empty is not empty: " << empty.isEmpty() << "\n";
	return 1;
    }
    CudaMatrix J2 = J + empty;
    if ( matrixCompare( J2.toHost(), j_lf, "J2 = J + empty") ) {
	return 1;
    }
    CudaMatrix J3 = empty - G;
    if ( matrixCompare( J3.toHost(), j_lf, "J3 = empty - G") ) {
	return 1;
    }
    CudaMatrix J4 = empty*G;
    if ( !J4.isEmpty() ) {
	std::cout << "J4 should be empty after J4 = empty*G, but it is not\n";
	return 1;
    }

    J3 = empty;
    if ( !J3.isEmpty() ) {
	std::cout << "J3 should be empty but is not, after J3 = empty\n";
	return 1;
    }

    // Create a diagonal matrix from CollOfScalar
    CollOfScalar k_cos(j_v);
    CudaMatrix K(k_cos);
    vector<int> k_rp = {0,1,2,3,4,5,6,7,8,9,10,11,12};
    vector<int> k_ci = {0,1,2,3,4,5,6,7,8,9,10,11};
    hostMat k_lf = { j_v, k_rp, k_ci, 12,12,12};
    if ( matrixCompare( K.toHost(), k_lf, "K(CollOfScalar)") ) {
	return 1;
    }

    // MATRIX * VECTOR OPERATION
    vector<double> L_ca_vec = {-3,4,2,1};
    CudaArray L_ca(L_ca_vec);
    CudaArray L = F*L_ca;
    vector<double> l_lf = {31, 1, -39.5, 4};
    if ( arrayCompare( L.copyToHost(), l_lf, "CudaArray L(CudaMatrix*CudaArray)") ) {
	return 1;
    }

    // Matrix transpose
    // Test by Matrix_transpose * Vector
    //vector<double> ft_v = {-1,4.5,3,-3,-1,8,-4,1,1,-6,6};
    //vector<int> ft_rp = {0,2,5,8,11};
    //vector<int> ft_ci = {0,2,0,2,3,0,2,3,1,2,3};
    //hostMat ft_lf = { ft_v, ft_rp, ft_ci, 11, 4, 4};
    CudaMatrix Ft = F.transpose();
    CudaArray FT_L = Ft * L_ca;
    vector<double> ft_l_lf = {12, -16, -31, -2};
    if ( arrayCompare( FT_L.copyToHost(), ft_l_lf, "F.transpose() * L_ca") ) {
	return 1;
    }

    // Create a big diagonal matrix from CollOfScalar
    vector<double> diag_vec;
    vector<int> diag_rp;
    vector<int> diag_ci;
    int diag_size = 1000000;
    for( int i = 0; i < diag_size; i++) {
	diag_vec.push_back(i*0.01);
	diag_rp.push_back(i);
	diag_ci.push_back(i);
    }
    diag_rp.push_back(diag_size);
    CollOfScalar diag_cos(diag_vec);
    CudaMatrix diag(diag_cos);
    hostMat diag_lf = {diag_vec, diag_rp, diag_ci, diag_size, diag_size, diag_size};
    if ( matrixCompare( diag.toHost(), diag_lf, "Big diagonal matrix") ) {
	return 1;
    }
			   
    // 4 by 4 diag:
    vector<double> small_v = {10,100,1000, 10000};
    vector<int> small_rp = {0,1,2,3,4};
    vector<int> small_ci = {0,1,2,3};
    vector<double> smallDiagMult_val = {120, 160, 200, 100, -500, 4500, -1000, -18000,
					5000, 40000, 40000, 0};
    CollOfScalar small_cos(small_v);
    CudaMatrix small(small_cos);
    CudaMatrix smallDiagMult = small * E;
    hostMat smallDiagMult_lf = {smallDiagMult_val, e_rp, e_ci, 12, 4, 4};
    if ( matrixCompare( smallDiagMult.toHost(), smallDiagMult_lf, "smallDiagMult") ) {
	return 1;
    }


    // Check the matrices from DeviceHelperOps:
    vector<double> grad_v = {-1,1,-1,1,-1,1, -1,1,-1,1,-1,1, -1,1,-1,1,-1,1,
			     -1,1,-1,1,-1,1,-1,1, -1,1,-1,1,-1,1,-1,1};
    vector<int> grad_rp = {0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34};
    vector<int> grad_ic = {0,1,1,2,2,3, 4,5,5,6,6,7, 8,9,9,10,10,11,
			   0,4,1,5,2,6,3,7, 4,8,5,9,6,10,7,11};
    hostMat grad_host = {grad_v, grad_rp, grad_ic, 34, 17, 12};
    if ( matrixCompare(er.getGradMatrix().toHost(), grad_host, "DeviceHelperOps::grad")){
	return 1;
    }
    
    vector<double> div_v = {1,1, -1,1,1, -1,1,1, -1,1, 1,-1,1, -1,1,-1,1, -1,1,-1,1, 
			    -1,-1,1, 1,-1, -1,1,-1, -1,1,-1, -1,-1};
    vector<int> div_rp = {0,2,5,8,10,13,17,21,24,26,29,32,34};
    vector<int> div_ci = {0,9, 0,1,10, 1,2,11, 2,12, 3,9,13, 3,4,10,14, 4,5,11,15,
			  5,12,16, 6,13, 6,7,14, 7,8,15, 8,16};
    hostMat div_host = {div_v, div_rp, div_ci, 34, 12, 17};
    if ( matrixCompare(er.getDivMatrix().toHost(), div_host, "DeviceHelperOps::div") ) {
	return 1;
    }

    vector<double> fulldiv_v = { -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, 
				 -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, -1,1,-1,1, 
				 -1,1,-1,1, -1,1,-1,1};
    vector<int> fulldiv_rp = {0,4,8,12,16,20,24,28,32,36,40,44,48};
    vector<int> fulldiv_ci = {0,1,15,19, 1,2,16,20, 2,3,17,21, 3,4,18,22,
			      5,6,19,23, 6,7,20,24, 7,8,21,25, 8,9,22,26,
			      10,11,23,27, 11,12,24,28, 12,13,25,29, 13,14,26,30};
    hostMat fulldiv_host = {fulldiv_v, fulldiv_rp, fulldiv_ci, 48, 12, 31};
    if ( matrixCompare(er.getFulldivMatrix().toHost(), fulldiv_host, "DeviceHelperOps::fulldiv") ) {
	return 1;
    }
    
    
    

    return 0;

} // main()


