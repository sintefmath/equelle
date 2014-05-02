#include <iostream>
#include <string>

#include <opm/core/utility/ErrorMacros.hpp>

#include "LinearSolver.hpp"
#include "CudaMatrix.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

#include <cusp/array1d.h>
#include <cusp/csr_matrix.h>
#include <cusp/precond/diagonal.h>
#include <cusp/krylov/bicgstab.h>
#include <cusp/krylov/cg.h>



using namespace equelleCUDA;

LinearSolver::LinearSolver(std::string solver,
			   std::string precond,
			   int maxit,
			   double tol) 
    : solver_(BiCGStab),
      precond_(DIAG),
      tol_(tol),
      maxit_(maxit)
{
    // Check solver
    if ( solver == "BiCGStab" ) {
	// This is default value. Do nothing.
    }
    else if ( solver == "CG" ) {
	solver_ = CG;
    }
    else {
	printLegalInput();
	OPM_THROW(std::runtime_error, "Illegal input " << solver << " for solver.");
    }

    // Check preconditioner
    if ( precond == "diagonal" ) {
	// This is default value. Do nothing.
    }
    else if ( precond == "none" ) {
	precond_ = NONE;
    }
    else {
	printLegalInput();
	OPM_THROW(std::runtime_error, "Illegal input " << precond << " for preconditioner.");
    }
}


// Print legal input to screen:
void LinearSolver::printLegalInput() const {
    std::cout << "\n\nThe following are legal input for solver:\n";
    std::cout << "\t - CG\n";
    std::cout << "\t - BiCGStab\n";
    std::cout << "Example: solver=BiCGStab\n";
    std::cout << "\n";
    std::cout << "The following are legal input for preconditioner:\n";
    std::cout << "\t - none\n";
    std::cout << "\t - diagonal\n";
    std::cout << "Example: preconditioner=diagonal\n";
    std::cout << "\n";
}

void LinearSolver::printLegalCombos() const {
    std::cout << "\nThe following combinations of solver and preconditioner";
    std::cout << "are implemented:\n";
    std::cout << "\tCG + none\n";
    std::cout << "\tCG + diagonal\n";
    std::cout << "\tBiCGStab + none\n";
    std::cout << "\tBiCGStab + diagonal (default)\n";
    std::cout << "Example use in parameter file:\n";
    std::cout << "\tsolver=CG\n";
    std::cout << "\tpreconditioner=diagonal\n";
}

LinearSolver::~LinearSolver() {
    // Nothing to do here
}


// Solver:
CollOfScalar LinearSolver::solve(const CudaMatrix& A_cpy, const CudaArray& b_cpy) const {
    
    // Check square matrix, and that A.rows() = b.size()
    // Need also to check which method and preconditioner to use.
   
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

    // Solve according to solver and preconditioner:
    if ( solver_ == BiCGStab && precond_ == DIAG ) {
	// create preconditioner
	cusp::precond::diagonal<double, cusp::device_memory> M(cusp_A);
	// Solve the system
	cusp::krylov::bicgstab(cusp_A, cusp_x, cusp_b, monitor, M);
    }
    else if ( solver_ == CG && precond_ == DIAG ) {
	cusp::precond::diagonal<double, cusp::device_memory> M(cusp_A);
	cusp::krylov::cg(cusp_A, cusp_x, cusp_b, monitor, M);
    }
    else if ( solver_ == BiCGStab && precond_ == NONE ) {
	cusp::krylov::bicgstab(cusp_A, cusp_x, cusp_b, monitor);
    }
    else if ( solver_ == CG && precond_ == NONE ) {
	cusp::krylov::cg(cusp_A, cusp_x, cusp_b, monitor);
    }
    else {
	printLegalCombos();
	OPM_THROW(std::runtime_error, "The given combination of solver and preconditioner is not implemented");
    }

    // Print info
    std::cout << "\n";
    std::cout << "cusp iterations: " << monitor.iteration_count() << "\n";  
    std::cout << "cusp residual norm: " << monitor.residual_norm() << "\n";
    std::cout << "\n";

    return CollOfScalar(x);
}
