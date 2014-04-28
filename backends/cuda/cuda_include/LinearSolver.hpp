
#ifndef EQUELLE_LINEARSOLVER_HEADER_INCLUDED
#define EQUELLE_LINEARSOLVER_HEADER_INCLUDED

#include "CudaArray.hpp"
#include "CudaMatrix.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

namespace equelleCUDA {

    
    class LinearSolver
    {
    public:
	LinearSolver();
	~LinearSolver();

	CollOfScalar solve(const CudaMatrix& A, const CudaArray& b);
	
    private:
	EquelleSolver solver;
	EquellePrecond precond;

    }; // class LinearSolver



} // namespace equelleCUDA


#endif // EQUELLE_LINEARSOLVER_HEADER_INCLUDED
