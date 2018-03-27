
#ifndef EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
#define EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED


#include <opm/grid/GridManager.hpp>
#include <opm/grid/UnstructuredGrid.h>

#include <Eigen/Sparse>
#include <Eigen/Eigen>
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
    class DeviceHelperOps {
	
    public:



	//! Constructor
	/*!
	  Input is Eigen sparse matrices as stored by the Opm::HelperOps class.
	*/
	DeviceHelperOps(const UnstructuredGrid& grid_in);

	//! Destructor
	~DeviceHelperOps() { };


	//! Extract for each face the difference of its adjacent cells' values (second - first)
	/*! 
	  Matrix size: rows = number of internal faces, cols = number of cells.
	*/
	const CudaMatrix& grad();
	//! Extract for each cell the sum of its adjacent interior faces' (signed) values.
	/*! 
	  Matrix size: rows = number of cells, cols = number of internal faces.
	*/
	const CudaMatrix& div();
	//! Extract for each cell the sum of all its adjacent faces' (signed) values.
	/*!
	  Matrix size: rows = number of cells, cols = number of faces.
	*/
	const CudaMatrix& fulldiv();

	//! Number of internal faces
	/*!
	  The Opm helper class got this information, and it will be useful for
	  error checking in the Divergence function.
	*/
	int num_int_faces();

    private:
	bool initialized_;

	CudaMatrix grad_;
	CudaMatrix div_;
	CudaMatrix fulldiv_;
	int num_int_faces_;

	const UnstructuredGrid& grid_;

	void initGrad_();
	void initDiv_();
	void initFulldiv_();



	typedef Eigen::SparseMatrix<double> M;

	/// A list of internal faces.
	typedef Eigen::Array<int, Eigen::Dynamic, 1> IFaces;
	IFaces host_internal_faces_;
	
	/// Extract for each internal face the difference of its adjacent cells' values (first - second).
	M host_ngrad_;
	/// Extract for each face the difference of its adjacent cells' values (second - first).
	M host_grad_;
	/// Extract for each face the average of its adjacent cells' values.
	M host_caver_;
	/// Extract for each cell the sum of its adjacent interior faces' (signed) values.
	M host_div_;
	/// Extract for each face the difference of its adjacent cells' values (first - second).
	/// For boundary faces, one of the entries per row (corresponding to the outside) is zero.
	M host_fullngrad_;
	/// Extract for each cell the sum of all its adjacent faces' (signed) values.
	M host_fulldiv_;

	void initHost_();
    };


} // namespace equelleCUDA


#endif // EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
