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
	
	CollOfScalar extendToFull( const CollOfScalar& inData, 
				   const thrust::device_vector<int>& from_set,
				   const int& full_size);

	CollOfScalar extendToSubset( const CollOfScalar& inData,
				     const thrust::device_vector<int>& from_set,
				     const thrust::device_vector<int>& to_set,
				     const int& full_size);
			
	__global__ void extendToFullKernel( double* outData,
					    const int* from_set,
					    const int from_size,
					    const double* inData,
					    const int to_size);





	CollOfScalar onFromFull( const CollOfScalar& inData,
				 const thrust::device_vector<int>& to_set);

	CollOfScalar onFromSubset( const CollOfScalar& inData,
				   const thrust::device_vector<int>& from_set,
				   const thrust::device_vector<int>& to_set,
				   const int& full_size);

	__global__ void onFromFullKernel( double* outData,
					  const int* to_set,
					  const int to_size,
					  const double* inData);
				 
	



	thrust::device_vector<int> onFromFullIndices( const thrust::device_vector<int>& inData,
						      const thrust::device_vector<int>& from_set);

	thrust::device_vector<int> onFromSubsetIndices( const thrust::device_vector<int>& inData,
							const thrust::device_vector<int>& from_set,
							const thrust::device_vector<int>& to_set,
							const int& full_size);

	__global__ void onFromFullKernelIndices( int* outData,
						 const int* to_set,
						 const int to_size,
						 const int* inData);

	// Needed from onFromSubsetIndices
	thrust::device_vector<int> extendToFullIndices( const thrust::device_vector<int>& inData,
							const thrust::device_vector<int>& to_set,
							const int& full_size);

	__global__ void extendToFullKernelIndices( int* outData,
						   const int* from_set,
						   const int from_size,
						   const int* inData,
						   const int to_size);

    } // namespace wrapDeviceGrid







} // namespace equelleCUDA

#endif // EQUELLE_WRAP_DEVICEGRID_HEADER_INCLUDED
