
#ifndef EQUELLE_LINEARSOLVER_HEADER_INCLUDED
#define EQUELLE_LINEARSOLVER_HEADER_INCLUDED

#include "CudaArray.hpp"
#include "CudaMatrix.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

namespace equelleCUDA {

    //! For finding implicit solutions
    /*!
      This class contains the implementation of the linear solver which is needed 
      for finding implicit solutions with the NewtonSolve Equelle function.
      
      We store the numerical method we want to use, as well as the choice of
      preconditioner. By default we use BiCGStab with a diagonal preconditioner.
      
      The solvers are from the CUSP library.

      In the future it would be fun to include a test function which helps the user
      find out which solver to use, and automatize this process.

      \sa EquelleSolver, EquellePrecond
    */
    class LinearSolver
    {
    public:
	//! Default constructor
	/*!
	  The default method s BiCGStab with diagonal preconditioner, precision 
	  1e-10 and a maximum iteration of 1000.
	*/
	LinearSolver();
	//! Destructor
	~LinearSolver();

	//! Solves A*x = b 
	/*!
	  Wraps the matrix A and vector b into the thrust and cusp libraries in order 
	  to solve the system.

	  Since the cusp methods do not accept constant input, we need to do an
	  internal copy inside this function.
	*/
	CollOfScalar solve(const CudaMatrix& A, const CudaArray& b) const;
	
    private:
	EquelleSolver solver_;
	EquellePrecond precond_;
	double tol_;
	int maxit_;

    }; // class LinearSolver



} // namespace equelleCUDA


#endif // EQUELLE_LINEARSOLVER_HEADER_INCLUDED
