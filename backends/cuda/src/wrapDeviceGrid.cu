#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include <opm/core/utility/ErrorMacros.hpp>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"

using namespace equelleCUDA;

// --------------------------------------------
//                     EXTEND
// --------------------------------------------

CollOfScalar wrapDeviceGrid::extendToFull( const CollOfScalar& in_data,
					   const thrust::device_vector<int>& from_set,
					   const int& full_size) {
    std::cout << "WRAPPER\n";
    // setup how many threads/blocks we need:
    dim3 block(MAX_THREADS);
    dim3 grid( (int)((full_size + MAX_THREADS - 1)/ MAX_THREADS) );
    
    // create a vector of size number_of_faces_:
    //thrust::device_vector<double> out(full_size);
    CollOfScalar out(full_size);
    //double* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const int* from_ptr = thrust::raw_pointer_cast( &from_set[0]);
    wrapDeviceGrid::extendToFullKernel<<<grid,block>>>( out.data(),
							from_ptr,
							from_set.size(),
							in_data.data(),
							full_size);
    
      
    return out;
}

CollOfScalar wrapDeviceGrid::extendToSubset( const CollOfScalar& inData,
					     const thrust::device_vector<int>& from_set,
					     const thrust::device_vector<int>& to_set,
					     const int& full_size) {
    std::cout << "WRAPPER - Extend to subset\n";
    CollOfScalar temp_full = extendToFull( inData, from_set, full_size);
    return onFromFull(temp_full, to_set);

}

__global__ void wrapDeviceGrid::extendToFullKernel( double* outData,
						    const int* from_set,
						    const int from_size,
						    const double* inData,
						    const int to_size) 
{
    int outIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if ( outIndex < to_size) {
	outData[outIndex] = 0;
	
	__syncthreads();
	if ( outIndex < from_size ) {
	    outData[from_set[outIndex]] = inData[outIndex];
	}
    }
}


// ------------------------------------------------
//                ON for CollOfScalar
// ------------------------------------------------

CollOfScalar wrapDeviceGrid::onFromFull( const CollOfScalar& inData,
					 const thrust::device_vector<int>& to_set ) {

    // inData is a full set, so position is its index
    // to_set is indices which we get the input from.
    // out will be same size as to_set.

    std::cout << "WRAPPER\n";
    // setup how many threads/blocks we need:
    dim3 block(MAX_THREADS);
    dim3 grid( (int)(( to_set.size() + MAX_THREADS - 1)/ MAX_THREADS) );
    
    // Create the output vector:
    CollOfScalar out(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    wrapDeviceGrid::onFromFullKernel<<<grid,block>>>(out.data(),
						     to_set_ptr,
						     to_set.size(),
						     inData.data());
    return out;
}

CollOfScalar wrapDeviceGrid::onFromSubset( const CollOfScalar& inData,
					   const thrust::device_vector<int>& from_set,
					   const thrust::device_vector<int>& to_set,
					   const int& full_size) {
    
    std::cout << "WRAPPER - On subset\n";
    CollOfScalar temp_full = extendToFull(inData, from_set, full_size);
    return onFromFull(temp_full, to_set);
}



__global__ void wrapDeviceGrid::onFromFullKernel( double* outData,
						  const int* to_set,
						  const int to_size,
						  const double* inData)
{
    int toIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if ( toIndex < to_size ) {
	outData[toIndex] = inData[to_set[toIndex]];
    }
}
						  

// -----------------------------------------------
//              ON for CollOfIndices
// -----------------------------------------------

thrust::device_vector<int> wrapDeviceGrid::onFromFullIndices( const thrust::device_vector<int>& inData,
							      const thrust::device_vector<int>& to_set ) {

    // inData is a full set, so position is its index
    // to_set is indices which we get the input from.
    // out will be same size as to_set.

    std::cout << "WRAPPER\n";
    // setup how many threads/blocks we need:
    dim3 block(MAX_THREADS);
    dim3 grid( (int)(( to_set.size() + MAX_THREADS - 1)/ MAX_THREADS) );
    
    // Create the output vector:
    thrust::device_vector<int> out(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    const int* inData_ptr = thrust::raw_pointer_cast( &inData[0] );
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    wrapDeviceGrid::onFromFullKernelIndices<<<grid,block>>>(out_ptr,
							    to_set_ptr,
							    to_set.size(),
							    inData_ptr);
    return out;
}



thrust::device_vector<int> wrapDeviceGrid::onFromSubsetIndices( const thrust::device_vector<int>& inData,
								const thrust::device_vector<int>& from_set,
								const thrust::device_vector<int>& to_set,
								const int& full_size) {
    
    std::cout << "WRAPPER - On subset\n";
    thrust::device_vector<int> temp_full = extendToFullIndices(inData, from_set, full_size);
    return onFromFullIndices(temp_full, to_set);
}




__global__ void wrapDeviceGrid::onFromFullKernelIndices( int* outData,
							 const int* to_set,
							 const int to_size,
							 const int* inData)
{
    int toIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if ( toIndex < to_size ) {
	outData[toIndex] = inData[to_set[toIndex]];
    }
}


thrust::device_vector<int> wrapDeviceGrid::extendToFullIndices( const thrust::device_vector<int>& in_data,
								const thrust::device_vector<int>& from_set,
								const int& full_size) {
    std::cout << "WRAPPER\n";
    // setup how many threads/blocks we need:
    dim3 block(MAX_THREADS);
    dim3 grid( (int)((full_size + MAX_THREADS - 1)/ MAX_THREADS) );
    
    // create a vector of size number_of_faces_:
    thrust::device_vector<int> out(full_size);
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const int* in_data_ptr = thrust::raw_pointer_cast( &in_data[0] );
    const int* from_ptr = thrust::raw_pointer_cast( &from_set[0]);
    wrapDeviceGrid::extendToFullKernelIndices<<<grid,block>>>( out_ptr,
							       from_ptr,
							       from_set.size(),
							       in_data_ptr,
							       full_size);
    
      
    return out;
}


__global__ void wrapDeviceGrid::extendToFullKernelIndices( int* outData,
							   const int* from_set,
							   const int from_size,
							   const int* inData,
							   const int to_size) 
{
    int outIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if ( outIndex < to_size) {
	outData[outIndex] = 0;
	
	__syncthreads();
	if ( outIndex < from_size ) {
	    outData[from_set[outIndex]] = inData[outIndex];
	}
    }
}

