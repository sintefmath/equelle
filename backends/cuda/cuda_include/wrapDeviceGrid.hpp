#ifndef EQUELLE_WRAP_DEVICEGRID_HEADER_INCLUDED
#define EQUELLE_WRAP_DEVICEGRID_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"


namespace equelleCUDA
{



    //! Functions that contain device code but that can not be directly part the class.
    /*!
      This namespace contains functions that is closely related to the DeviceGrid
      class, but that can not be included in the class itself. Functions here are 
      often called from template functions, or are kernels.
      
      The functions here should be thought of as private class members. They are
      available as regular functions, but they are designed to be called from other 
      higher level functions in either the equelleCUDA classes or the EquelleRuntimeCUDA
      class.
    */
    namespace wrapDeviceGrid{
	
	//! Extend from a subset to either AllCells() or AllFaces()
	/*!
	  This function takes a collection of Scalar defined on the subset of
	  indices given in from_set and maps them to the same indices for the
	  complete grid.
	  
	  \param[in] inData The set of scalars that should be extended.
	  \param[in] from_set The indices in which each element in the inData lives.
	  \param[in] full_size The size of all grid entities of the given type in 
	  the grid. This will be the number of cells or number of faces.

	  \return The collection of Scalars such that the elements in the from_set
	  has the values found in inData and the rest is zero.
	*/
	CollOfScalar extendToFull( const CollOfScalar& inData, 
				   const thrust::device_vector<int>& from_set,
				   const int full_size);
	
	//! Extend a Collection of Scalars from a subset to another subset in the grid.
	/*!
	  Extend from a subset A to another subset B, where A is also a subset of B.
	  This implementation simply consist of two function calls. First, the collection
	  of Scalars are extended to the entire set of same grid entities, then the 
	  complete set is restricted to the subset B.
	  
	  \param[in] inData The Collection of Scalars that should be extended.
	  \param[in] from_set Indices to be extended from
	  \param[in] to_set Indices in the resulting set.
	  \param[in] full_size The complete number of grid entities (faces or cells) 
	  associated with this set.

	  \return The Collection of Scalars such that the elements on the from_set
	  indices are the values found in inData, and the other indices are zero.
	*/
	CollOfScalar extendToSubset( const CollOfScalar& inData,
				     const thrust::device_vector<int>& from_set,
				     const thrust::device_vector<int>& to_set,
				     const int full_size);
			


	//! Kernel for extend a subset to a full set - step 1
	/*!
	  This operation maps the data from inData to the indices given in from_set
	  to outData. The elements in outData that has not any corresponding 
	  elements in the inData are given the value zero.
	  Since the outData array has to be a complete set, the integer values
	  in from_set match the index of outData.
	  
	  The operation has to be done in two steps, as we need full synchronization
	  between all blocks to avoid race conditions. Since __syncthreads() only
	  synchronize threads in the same block, we have to synchronize by 
	  having two kernels.

	  Step 1 - Done by this kernel:
	  \code
	  For each i = 0:to_size-1
	       outData[i] = 0
	  \endcode
	  Step 2:
	  \code
	  For each i = 0:from_size-1
	       outData[from_set[i]] = inData[i]
	  \endcode
	  
	  \param[out] outData Are to be filled with zeroes 
	  \param[in] fullSize Size of outData.

	  \sa extendToFullKernel_step2
	*/
	__global__ void extendToFullKernel_step1( double* outData,
						  const int fullSize);

	//! Kernel for extend a subset to a full set - step 2
	/*!
	  This operation maps the data from inData to the indices given in from_set
	  to outData. The elements in outData that has not any corresponding 
	  elements in the inData are given the value zero.
	  Since the outData array has to be a complete set, the integer values
	  in from_set match the index of outData.
	  
	  The operation has to be done in two steps, as we need full synchronization
	  between all blocks to avoid race conditions. Since __syncthreads() only
	  synchronize threads in the same block, we have to synchronize by 
	  having two kernels.

	  Step 1:
	  \code
	  For each i = 0:to_size-1
	       outData[i] = 0
	  \endcode
	  Step 2 - Done by this kernel:
	  \code
	  For each i = 0:from_size-1
	       outData[from_set[i]] = inData[i]
	  \endcode
	  
	  \param[in,out] outData The extended set of doubles. Are initialized with
	  zeroes on input.
	  \param[in] from_set Indices for each of the input variables when 
	  mapped to the complete set.
	  \param[in] from_size Size of from_set and inData.
	  \param[in] inData The array that should be extended.

	  \sa extendToFullKernel_step1
	*/
	__global__ void extendToFullKernel_step2( double* outData,
						  const int* from_set,
						  const int from_size,
						  const double* inData);


	//! On operator from a complete domain to a subset.
	/*!
	  Given a Collection of Scalars defined on either AllCells or AllFaces,
	  this function return a Collection of Scalars consisting of only the 
	  values found in the indices given in to_set.

	  \param[in] inData Collection of Scalars defined on a complete set in the grid.
	  \param[in] to_set Indices which we want to create a restriction from.
	  
	  \return A Collection of Scalars which consists of the values in inData on the 
	  indices found in to_set.
	*/
	CollOfScalar onFromFull( const CollOfScalar& inData,
				 const thrust::device_vector<int>& to_set);

	//! On operator maping between two different subsets of the grid.
	/*!
	  Given two vectors of indices which creates two subsets of the grid,
	  we want to return a Collection of Scalars of the values found in inData 
	  which where the indices of to_set matches from_set.

	  This function is implemented by two function calls only. First, a 
	  temp collection is found by extending inData to the full set, then
	  the return value is found by mapping the temp set from to_set by
	  calling the onFromFullKernel.
	  
	  \param[in] inData Collection of Scalars which we create a subset of.
	  \param[in] from_set The indices each element of inData lives on.
	  \param[in] to_set The indices we want to have in the output.
	  \param[in] full_size The number of total entities of this type in the grid.
	  If the two index sets are cell indices this will be number_of_cells_ and if 
	  we have face indices, this will be number_of_faces.

	  \return A Collection of Scalars found in inData associated with the indices 
	  found in to_set.
	*/
	CollOfScalar onFromSubset( const CollOfScalar& inData,
				   const thrust::device_vector<int>& from_set,
				   const thrust::device_vector<int>& to_set,
				   const int full_size);

	//! Kernel for performing On from a complete set to a subset.
	/*!
	  This kernel stores in outData the values found in inData associated with
	  the indices in to_set. 

	  Performs the following pseodocode:
	  \code
	  For each i = 0:to_size-1
	      outData[i] = inData[to_set[i]]
	  \endcode

	  \param[in,out] outData The output values that becomes a Collection of Scalars.
	  \param[in] to_set The indices we from inData in the outData.
	  \param[in] to_size The size of to_set and outData.
	  \param[in] inData The set from which we create the output.
	*/
	__global__ void onFromFullKernel( double* outData,
					  const int* to_set,
					  const int to_size,
					  const double* inData);
				 


	// ---------- INDICES --------------------- // 
	

	//! On operator for input and output as Collection of Indices. Here for input as a full set.
	/*!
	  The On operator in equelle can be used as part of an evaluate-on
	  operation as well as a restrict-to operator. The implementation is 
	  exactly the same as for the case of Collection of Scalars, but with
	  other data types. Here the input set is defined on all entities in the grid.
	  
	  \param[in] inData A set of indices defined on a complete set in the grid.
	  \param[in] to_set The indices in inData we want to give as output.

	  \return A set of indices from inData corresponding to the to_set.

	  \sa wrapDeviceGrid::onFromFull
	*/
	thrust::device_vector<int> onFromFullIndices( const thrust::device_vector<int>& inData,
						      const thrust::device_vector<int>& to_set);
	
	//! On operator for Collection of Indices from subset to subset.
	/*!
	  The On operator in equelle can be used as part of an evaluate-on
	  operation as well as a restrict-to operator. The implementation is 
	  exactly the same as for the case of Collection of Scalars, but with
	  other data types. Here the input set is defined on a subset in the grid.
	  
	  \param[in] inData A set of indices defined on a subset in the grid.
	  \param[in] from_set The subset of the grid where inData is defined.
	  \param[in] to_set The indices in inData we want to give as output.
	  \param[in] full_size The complete size of this kind of entities.

	  \return A set of indices from inData corresponding to the to_set.

	  \sa wrapDeviceGrid::onFromSubset
	*/
	thrust::device_vector<int> onFromSubsetIndices( const thrust::device_vector<int>& inData,
							const thrust::device_vector<int>& from_set,
							const thrust::device_vector<int>& to_set,
							const int full_size);

	//! Kernel to perform the On operation for Collection of Indices.
	/*!
	  This kernel stores in outData the values found in inData associated with
	  the indices in to_set. 

	  Performs the following pseodocode:
	  \code
	  For each i = 0:to_size-1
	      outData[i] = inData[to_set[i]]
	  \endcode

	  \param[in,out] outData The output values that becomes a Collection of Indices
	  \param[in] to_set The indices we from inData in the outData.
	  \param[in] to_size The size of to_set and outData.
	  \param[in] inData The set from which we create the output.
	  
	  \sa wrapDeviceGrid::onFromFullKernel.
	*/
	__global__ void onFromFullKernelIndices( int* outData,
						 const int* to_set,
						 const int to_size,
						 const int* inData);

	//! Extend a subset to a full set for Indices.
	/*!
	  This function takes a vector with indices defined on the subset of
	  indices given in from_set and maps them to the same indices for the
	  complete grid.

	  This function is never used stand-alone, but only to create a 
	  temporary result within onFromSubsetIndices.
	  
	  \param[in] inData The vector of indices that should be extended.
	  \param[in] to_set Indices in the resulting set.
	  \param[in] full_size The size of the complete set.
	
	  \return A vector with indices such that the elements on the from_set
	  indices are the values found in inData, and the other indices are zero.
	  
	  \sa wrapDeviceGrid::extendToFull, wrapDeviceGrid::onFromSubsetIndices
	*/
	thrust::device_vector<int> extendToFullIndices( const thrust::device_vector<int>& inData,
							const thrust::device_vector<int>& to_set,
							const int full_size);



	//! Kernel for extend a subset to a full set for Indices - step 1
	/*!
	  This operation maps the data from inData to the indices given in from_set
	  to outData. The elements in outData that has not any corresponding 
	  elements in the inData are given the value zero.
	  Since the outData array has to be a complete set, the integer values
	  in from_set match the index of outData.
	  
	  The operation has to be done in two steps, as we need full synchronization
	  between all blocks to avoid race conditions. Since __syncthreads() only
	  synchronize threads in the same block, we have to synchronize by 
	  having two kernels.

	  Step 1 - Done by this kernel:
	  \code
	  For each i = 0:to_size-1
	       outData[i] = 0
	  \endcode
	  Step 2:
	  \code
	  For each i = 0:from_size-1
	       outData[from_set[i]] = inData[i]
	  \endcode

	  This function is never used stand-alone, but only to create a 
	  temporary result within onFromSubsetIndices. Hence, the meaningless
	  operation by using zero (a legal index) as a fill-in, will be ignored 
	  by the onFromSubsetIndices function.
	  
	  \param[out] outData Output to be filled with zeros.
	  \param[in] full_size Size of outData array.

	  \sa extendToFullKernelIndices_step2
	*/
	__global__ void extendToFullKernelIndices_step1( int* outData,
							 const int full_size);

	//! Kernel for Extend a subset to a full set for Indices - step 2
	/*!
	  This operation maps the data from inData to the indices given in from_set
	  to outData. The elements in outData that has not any corresponding 
	  elements in the inData are given the value zero.
	  Since the outData array has to be a complete set, the integer values
	  in from_set match the index of outData.
	  	  
	  The operation has to be done in two steps, as we need full synchronization
	  between all blocks to avoid race conditions. Since __syncthreads() only
	  synchronize threads in the same block, we have to synchronize by 
	  having two kernels.
	  
	  Step 1:
	  \code
	  For each i = 0:to_size-1
	       outData[i] = 0
	  \endcode
	  Step 2 - Done by this kernel:
	  \code
	  For each i = 0:from_size-1
	       outData[from_set[i]] = inData[i]
	  \endcode

	  This function is never used stand-alone, but only to create a 
	  temporary result within onFromSubsetIndices. Hence, the meaningless
	  operation by using zero (a legal index) as a fill-in, will be ignored 
	  by the onFromSubsetIndices function.
	  
	  \param[in,out] outData The extended set of integers. Initialized as 
	  zeroes on input.
	  \param[in] from_set Indices for each of the input variables when 
	  mapped to the complete set.
	  \param[in] from_size Size of from_set and inData.
	  \param[in] inData The array that should be extended.

	  /sa extendToFullKernelIndices_step1, wrapDeviceGrid::extendToFull, wrapDeviceGrid::onFromSubsetIndices
	*/
	__global__ void extendToFullKernelIndices_step2( int* outData,
							 const int* from_set,
							 const int from_size,
							 const int* inData);


    } // namespace wrapDeviceGrid







} // namespace equelleCUDA

#endif // EQUELLE_WRAP_DEVICEGRID_HEADER_INCLUDED
