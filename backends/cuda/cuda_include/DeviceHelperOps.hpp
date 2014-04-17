
#ifndef EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
#define EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED


//#include <opm/autodiff/AutoDiffHelpers.hpp>

#include <Eigen/Sparse>
#include <iostream>

#include "CudaMatrix.hpp"


namespace equelleCUDA {

    //! Holds matrices needed for automatic differentiations for grid operations.
    /*!
      This is a GPU version of a subset of the HelperOps class from the
      opm/autodiff library. It is used so that OPM builds the matrices before the
      initialization of the DeviceHelperOps. This struct therefore holds device copies
      of some of the matrices created by OPM.
    */
    struct DeviceHelperOps {
	
	//! Extract for each face the difference of its adjacent cells' values (second - first)
	/*! 
	  Matrix size: rows = number of internal faces, cols = number of cells.
	*/
	CudaMatrix grad;
	//! Extract for each cell the sum of its adjacent interior faces' (signed) values.
	/*! 
	  Matrix size: rows = number of cells, cols = number of internal faces.
	*/
	CudaMatrix div;
	//! Extract for each cell the sum of all its adjacent faces' (signed) values.
	/*!
	  Matrix size: rows = number of cells, cols = number of faces.
	*/
	CudaMatrix fulldiv;

	//! Constructor
	/*!
	  Input is Eigen sparse matrices as stored by the Opm::HelperOps class.
	*/
	DeviceHelperOps( const Eigen::SparseMatrix<Scalar>& hostGrad, 
			 const Eigen::SparseMatrix<Scalar>& hostDiv, 
			 const Eigen::SparseMatrix<Scalar>& hostFullDiv)
	    : grad(hostGrad),
	      div(hostDiv),
	      fulldiv(hostFullDiv)
	{ };

	//! Destructor
	~DeviceHelperOps() { };

    };


} // namespace equelleCUDA


#endif // EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
