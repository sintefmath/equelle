#include <iostream>

#include "LinearSolver.hpp"
#include "CudaMatrix.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/precond/diagonal.h>
#include <cusp/krylov/bicgstab.h>



using namespace equelleCUDA;

LinearSolver::LinearSolver() 
    : solver_(BiCGStab),
      precond_(DIAG),
      tol_(1e-10),
      maxit_(1000)
{
    // intentionally left empty
}

LinearSolver::~LinearSolver() {
    // Nothing to do here
}


// Solver:
CollOfScalar LinearSolver::solve(const CudaMatrix& A_cpy, const CudaArray& b_cpy) {
    
    // Check square matrix, and that A.rows() = b.size()
   
    // hack to get around const issues...
    CudaMatrix A = A_cpy;
    CudaArray b = b_cpy; 

    // Declare the output.
    CudaArray x(b.size());

    // Handy typedefs:
    typedef typename thrust::device_ptr<double> double_ptr;
    typedef typename thrust::device_ptr<int> int_ptr;
    typedef typename cusp::array1d_view<int_ptr> intArrayView;
    typedef typename cusp::array1d_view<double_ptr> doubleArrayView;
    typedef typename cusp::csr_matrix_view<intArrayView, intArrayView, doubleArrayView> matrixView;
    // These creates const issues when we get to the solver...
    /*
    typedef typename thrust::device_ptr<const double> double_ptr;
    typedef typename thrust::device_ptr<double> mutable_double_ptr;
    typedef typename thrust::device_ptr<const int> int_ptr;
    typedef typename cusp::array1d_view<int_ptr> intArrayView;
    typedef typename cusp::array1d_view<double_ptr> doubleArrayView;
    typedef typename cusp::array1d_view<mutable_double_ptr> mutable_doubleArrayView;
    typedef typename cusp::csr_matrix_view<intArrayView, intArrayView, doubleArrayView> matrixView;
    */

    // Wrap the vectors and matrix into thrust
    double_ptr ptr_b(b.data());
    //mutable_double_ptr ptr_x(x.data());
    double_ptr ptr_x(x.data());

    double_ptr ptr_A_csrVal(A.csrVal());
    int_ptr ptr_A_csrRowPtr(A.csrRowPtr());
    int_ptr ptr_A_csrColInd(A.csrColInd());

    // Wrap the vectors and matrix into cusp
    doubleArrayView cusp_b(ptr_b, ptr_b + b.size());
    doubleArrayView cusp_x(ptr_x, ptr_x + x.size());
    
    doubleArrayView cusp_A_csrVal(ptr_A_csrVal, ptr_A_csrVal + A.nnz());
    intArrayView cusp_A_csrRowPtr(ptr_A_csrRowPtr, ptr_A_csrRowPtr + A.rows() + 1);
    intArrayView cusp_A_csrColInd(ptr_A_csrColInd, ptr_A_csrColInd + A.nnz());
    matrixView cusp_A( A.rows(), A.rows(), A.nnz(),
		       cusp_A_csrRowPtr, cusp_A_csrColInd, cusp_A_csrVal );

    // Create a monitor
    cusp::default_monitor<double> monitor(cusp_b, maxit_, tol_);

    // create preconditioner
    cusp::precond::diagonal<double, cusp::device_memory> M(cusp_A);

    // Solve the system
    cusp::krylov::bicgstab(cusp_A, cusp_x, cusp_b, monitor, M);

    // Print info
    std::cout << "\n";
    std::cout << "cusp iterations: " << monitor.iteration_count() << "\n";  
    std::cout << "cusp residual norm: " << monitor.residual_norm() << "\n";
    std::cout << "\n";

    return CollOfScalar(x);
}
