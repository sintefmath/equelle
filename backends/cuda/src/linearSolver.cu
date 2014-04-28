#include "LinearSolver.hpp"
#include "CudaMatrix.hpp"
#include "CudaArray.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"


using namespace equelleCUDA;

LinearSolver::LinearSolver() 
    : solver(BiCGStab),
      precond(DIAG)
{
    // intentionally left empty
}

LinearSolver::~LinearSolver() {
    // Nothing to do here
}


// Solver:
CollOfScalar LinearSolver::solve(const CudaMatrix& A, const CudaArray& b) {
    return CollOfScalar(b);
}
