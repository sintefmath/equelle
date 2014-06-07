
#ifndef EQUELLE_LINEARSOLVER_HEADER_INCLUDED
#define EQUELLE_LINEARSOLVER_HEADER_INCLUDED

#include <string>

#include "CudaArray.hpp"
#include "CudaMatrix.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

namespace equelleCUDA {


    //! Enumerator for specifying available linear solvers
    enum EquelleSolver { CG, BiCGStab, GMRes, CPU};

    //! Enumerator for specifying available preconditioners for linear solvers.
    enum EquellePrecond { NONE, DIAG};


    //! For finding implicit solutions
    /*!
      This class contains the implementation of the linear solver which is needed 
      for finding implicit solutions with the NewtonSolve Equelle function.
      
      We store the numerical method we want to use, as well as the choice of
      preconditioner. By default we use BiCGStab with a diagonal preconditioner.
      
      The solvers are from the CUSP library.

      In the future it would be fun to include a test function which helps the user
      find out which solver to use, and automatize this process.

      For more flexibility, we also allow the CPU option for solver, and we then 
      use the same solvers as in the Equelle CPU back-end.

      \note
      To the developer: In order to implement new solver/preconditioner combination:
      - Edit enum options to EquelleSolver and EquellePrecond.
      - Edit constructor to be able to accept the new method/precond
      - Implement new solver in the solve function
      - Edit printLegalInput() to print the new alternative
      - Edit printLegalCombos() to print the new alternative
      
      \todo
      Get rid of all if else options in function solve. Hide options in private 
      member functions to avoid the potential huge amount of combinations. Should
      check for preconditioner only once, then solver only once.

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
	LinearSolver(std::string solver, std::string precond,
		     int maxit, double tol);
	//! Destructor
	~LinearSolver();

	//! Solves A*x = b 
	/*!
	  Wraps the matrix A and vector b into the thrust and cusp libraries in order 
	  to solve the system.

	  Since the cusp methods do not accept constant input, we need to do an
	  internal copy inside this function.

	  For no output to screen: verbose = 0
	*/
	CollOfScalar solve(const CudaMatrix& A,
			   const CudaArray& b,
			   const int verbose ) const;
	
	//! Gives the solver enum value
	EquelleSolver getSolver() const;

    private:
	EquelleSolver solver_;
	EquellePrecond precond_;
	double tol_;
	int maxit_;

	//! Error messages
	/*!
	  In case the requested solver or preconditioner is not legal, this function
	  will be called and prints to screen the legal alternatives.
	*/
	void printLegalInput() const ;
	//! Error messages
	/*! 
	  If the requested combination of solver and preconditioner is not implemented
	  this function will be called and prints to screen the legal options for
	  solver-preconditioner combos.
	*/
	void printLegalCombos() const ;

    }; // class LinearSolver



} // namespace equelleCUDA


#endif // EQUELLE_LINEARSOLVER_HEADER_INCLUDED
