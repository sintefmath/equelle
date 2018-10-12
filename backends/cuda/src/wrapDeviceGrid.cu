#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include <opm/common/ErrorMacros.hpp>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "device_functions.cuh"

using namespace equelleCUDA;

// --------------------------------------------
//                     EXTEND
// --------------------------------------------

CollOfScalar wrapDeviceGrid::extendToFull( const CollOfScalar& in_data,
                       const thrust::device_vector<int>& from_set,
                       const int full_size)
{
    // Create a vector of size number_of_faces_:
    CudaArray val(full_size);
    // Extend values
    thrust::fill(thrust::device, val.data(), val.data()+full_size, 0.0);
    thrust::scatter(thrust::device, in_data.data(), in_data.data()+in_data.size(), from_set.begin(), val.data());
    if (in_data.useAutoDiff() ) {
        // Set up output matrix der
        CudaMatrix der(full_size, in_data.der_.cols(), in_data.der_.nnz());

        // Copy csrColInd, csrVal and fill csrRowPtr with zeroes
        thrust::copy(thrust::device, in_data.der_.csrColInd(), in_data.der_.csrColInd()+in_data.der_.nnz(), der.csrColInd());
        thrust::copy(thrust::device, in_data.der_.csrVal(), in_data.der_.csrVal()+in_data.der_.nnz(), der.csrVal());
        thrust::fill(thrust::device,der.csrRowPtr(),der.csrRowPtr()+der.rows()+1, 0.0);
        cudaDeviceSynchronize();
        // Map values in set being extended to the new domain
        thrust::scatter(thrust::device, in_data.der_.csrRowPtr()+1, in_data.der_.csrRowPtr()+in_data.der_.rows()+1, from_set.begin(), der.csrRowPtr()+1);
        cudaDeviceSynchronize();

        // Fill in the gaps of the rowPtr
        // {0, 0, 2, 0, 0, 4, 0, 5} becomes 
        // {0, 0, 2, 2, 2, 4, 4, 5}
        thrust::maximum<int> binary_op;
        thrust::inclusive_scan(thrust::device, der.csrRowPtr(), der.csrRowPtr()+der.rows()+1, der.csrRowPtr(), binary_op);
        cudaDeviceSynchronize();
        return CollOfScalar(val, der);
    }
    return CollOfScalar(val);
}

CollOfScalar wrapDeviceGrid::extendToSubset( const CollOfScalar& inData,
					     const thrust::device_vector<int>& from_set,
					     const thrust::device_vector<int>& to_set,
					     const int full_size) {
    CollOfScalar temp_full = extendToFull( inData, from_set, full_size);
    return onFromFull(temp_full, to_set);

}

__global__ void wrapDeviceGrid::extendToFullKernel_step1( double* outData,
							  const int out_size)
{
    const int outIndex = myID();
    if ( outIndex < out_size ) {
	outData[outIndex] = 0;
    }
}

__global__ void wrapDeviceGrid::extendToFullKernel_step2( double* outData,
							  const int* from_set,
							  const int from_size,
							  const double* inData)
{
    //
    //      This kernel is sensitive to a race condition!
    //      Each thread with outIndex < from_size performs 2 write operations,
    //      but not to the same memory.
    //      Hence, the we can have a kernel with 
    //          outIndex = 3;
    //	  outData[3] = 0;
    //	  from_set[3] = 1000;
    //	  outData[1000] = 3.14;
    //     And then another block starting a bit later with
    //         outIndex = 1000;
    //	 outData[1000] = 0; // overwriting outIndex(3)'s correct value
    //
    //	 Only way to sync between blocks is to call seperate kernels!
    //

    const int outIndex = myID();
    if ( outIndex < from_size ) {
	outData[from_set[outIndex]] = inData[outIndex];
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

    // setup how many threads/blocks we need:
    kernelSetup s(to_set.size());

    // Create the output vector:
    CudaArray val(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    wrapDeviceGrid::onFromFullKernel<<<s.grid, s.block>>>(val.data(),
							  to_set_ptr,
							  to_set.size(),
							  inData.data());
    if ( inData.useAutoDiff() ) {
	CudaMatrix onMatrix(to_set, inData.size());
	return CollOfScalar(val, onMatrix * inData.derivative());
    }
    else { // no AutoDiff
	return CollOfScalar(val);
    }    

    // Use the matrix and find the result from Matrix-vector multiplication
    //CudaMatrix onMatrix(to_set, inData.size());
    //return onMatrix * inData;
}

CollOfScalar wrapDeviceGrid::onFromSubset( const CollOfScalar& inData,
					   const thrust::device_vector<int>& from_set,
					   const thrust::device_vector<int>& to_set,
					   const int full_size) {
    
    CollOfScalar temp_full = extendToFull(inData, from_set, full_size);
    return onFromFull(temp_full, to_set);
}



__global__ void wrapDeviceGrid::onFromFullKernel( double* outData,
						  const int* to_set,
						  const int to_size,
						  const double* inData)
{
    const int toIndex = myID();
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

    // setup how many threads/blocks we need:
    kernelSetup s(to_set.size());

    // Create the output vector:
    thrust::device_vector<int> out(to_set.size());
    const int* to_set_ptr = thrust::raw_pointer_cast( &to_set[0] );
    const int* inData_ptr = thrust::raw_pointer_cast( &inData[0] );
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    wrapDeviceGrid::onFromFullKernelIndices<<<s.grid, s.block>>>(out_ptr,
								 to_set_ptr,
								 to_set.size(),
								 inData_ptr);
    return out;
}



thrust::device_vector<int> wrapDeviceGrid::onFromSubsetIndices( const thrust::device_vector<int>& inData,
								const thrust::device_vector<int>& from_set,
								const thrust::device_vector<int>& to_set,
								const int full_size) {
    
    thrust::device_vector<int> temp_full = extendToFullIndices(inData, from_set, full_size);
    return onFromFullIndices(temp_full, to_set);
}




__global__ void wrapDeviceGrid::onFromFullKernelIndices( int* outData,
							 const int* to_set,
							 const int to_size,
							 const int* inData)
{
    const int toIndex = myID();
    if ( toIndex < to_size ) {
	outData[toIndex] = inData[to_set[toIndex]];
    }
}


thrust::device_vector<int> wrapDeviceGrid::extendToFullIndices( const thrust::device_vector<int>& in_data,
								const thrust::device_vector<int>& from_set,
								const int full_size) {
    // setup how many threads/blocks we need:
    kernelSetup s(full_size);

    // create a vector of size number_of_faces_:
    thrust::device_vector<int> out(full_size);
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const int* in_data_ptr = thrust::raw_pointer_cast( &in_data[0] );
    const int* from_ptr = thrust::raw_pointer_cast( &from_set[0]);
    wrapDeviceGrid::extendToFullKernelIndices_step1<<<s.grid, s.block>>>( out_ptr,
									  full_size);
    wrapDeviceGrid::extendToFullKernelIndices_step2<<<s.grid, s.block>>>( out_ptr,
									  from_ptr,
									  from_set.size(),
									  in_data_ptr);
    
      
    return out;
}



// EXTEND TO FULL FOR INDICES DONE IN 2 STEPS

__global__ void wrapDeviceGrid::extendToFullKernelIndices_step1( int* outData,
								 const int full_size)
{
    const int outIndex = myID();
    if ( outIndex < full_size) {
	outData[outIndex] = 0;
    }
}

__global__ void wrapDeviceGrid::extendToFullKernelIndices_step2( int* outData,
								 const int* from_set,
								 const int from_size,
								 const int* inData)
{
    const int outIndex = myID();
    if ( outIndex < from_size ) {
	outData[from_set[outIndex]] = inData[outIndex];
    }
}

