//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/fill.h>
//#include <thrust/copy.h>
//#include <thrust/sequence.h>

#include <string>
//#include <fstream>
//#include <iterator>
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdlib.h>
#include <vector>
#include <iostream>

// For error exception macro:
#include <opm/core/utility/ErrorMacros.hpp>

#include "CudaArray.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"


// Implementation of the class CudaArray

using namespace equelleCUDA;
using namespace wrapCudaArray;

CudaArray::CudaArray() 
    : size_(0), 
      dev_values_(0),
      block_x_(0),
      grid_x_(0),
      setup_(0)
#ifdef EQUELLE_DEBUG
    , debug_vec_(0)
#endif // EQUELLE_DEBUG
{
    // Intentionally left blank
}


// Allocating memory without initialization
CudaArray::CudaArray(const int size) 
    : size_(size),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_),
      setup_(size_)
#ifdef EQUELLE_DEBUG
    , debug_vec_(size,0)
#endif // EQUELLE_DEBUG
{
    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CudaArray::CudaArray(int)");
#ifdef EQUELLE_DEBUG
    std::cout << "Debug mode is on!\n";
#endif // EQUELLE_DEBUG
}

CudaArray::CudaArray(const int size, const double value) 
    : size_(size),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_),
      setup_(size_)
#ifdef EQUELLE_DEBUG
    , debug_vec_(size, value)
#endif // EQUELLE_DEBUG
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.

    std::vector<double> host_vec(size_, value);

    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CudaArray::CudaArray(int, double)");
        
    cudaStatus_ = cudaMemcpy(dev_values_, &host_vec[0], size_*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy in CudaArray::CudaArray(int, double)");

} 


// Constructor from vector, in order to do testing
CudaArray::CudaArray(const std::vector<double>& host_vec)
    : size_(host_vec.size()),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_),
      setup_(size_)
#ifdef EQUELLE_DEBUG
    , debug_vec_(host_vec)
#endif // EQUELLE_DEBUG
{
    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CudaArray::CudaArray(std::vector<double>)");
    
    cudaStatus_ = cudaMemcpy(dev_values_, &host_vec[0], size_*sizeof(double),
			    cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy in CudaArray::CudaArray(std::vector<double>)");
}


// Copy constructor
CudaArray::CudaArray(const CudaArray& coll) 
    : size_(coll.size_), 
      dev_values_(0),
      grid_x_(coll.grid_x_),
      block_x_(coll.block_x_),
      setup_(size_)
#ifdef EQUELLE_DEBUG
    , debug_vec_(coll.size_, 0)
#endif // EQUELLE_DEBUG
{
    std::cout << __PRETTY_FUNCTION__ << std::endl;

    if (coll.dev_values_ != 0) {
	cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
	checkError_("cudaMalloc in CudaArray::CudaArray(const CudaArray&)"); 

	cudaStatus_ = cudaMemcpy(dev_values_, coll.dev_values_, size_*sizeof(double),
				 cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy in CudaArray::CudaArray(const CudaArray&)");
    }
    
#ifdef EQUELLE_DEBUG
    // Copy value to the std::vector debug_vec_
    std::cout << "\tDEBUG IS ON!\n";
    if (coll.dev_values_ != 0 ) {
	cudaStatus_ = cudaMemcpy( &debug_vec_[0], dev_values_, size_*sizeof(double),
				  cudaMemcpyDeviceToHost );
	checkError_("cudaMemcpy for DEBUG in CudaArray::CudaArray(const CudaArray&)");
	last_val = debug_vec_[size_ - 1];
    }
#endif // EQUELLE_DEBUG
}


// Copy assignment operator
CudaArray& CudaArray::operator= (const CudaArray& other) {
    std::cout << __PRETTY_FUNCTION__ << std::endl;

    // Protect agains " var = var " , self assignment
    if ( this != &other ) {

	// First idea: Make this->dev_values_ point to other.dev_values_
	// and set other.dev_values_ = 0.
	// Why is this a bad idea? We should still be able to use other.
	// THEREFORE: Need to copy the content of other.dev_values_ to
	// this->dev_values_.

	// this->dev_values will be overwritten, and can safely be freed,
	// But if the collections are of the same size (likely) we
	// Will just overwrite the old values.

	if ( this->size_ != other.size_) {

	    // If different size: Is this even allowed?
	    // Free memory:
	    cudaStatus_ = cudaFree(this->dev_values_);
	    checkError_("cudaFree(this->dev_values_) in CudaArray::operator=(const CudaArray&)");
	    // Allocate new memory:
	    cudaStatus_ = cudaMalloc((void**)&this->dev_values_,
				     sizeof(double) * other.size_);
	    checkError_("cudaMalloc(this->dev_values_) in CudaArray::operator=(const CudaArray&)");
	    
	    // Set variables depending on size_:
	    this->size_ = other.size_;
	    this->block_x_ = other.block_x_;
	    this->grid_x_ = other.grid_x_;
	}

	// Copy memory block from other to this:
	cudaStatus_ = cudaMemcpy( this->dev_values_, other.dev_values_,
				  sizeof(double) * this->size_,
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(dev_values_) in CudaArray::operator=(const CudaArray&)");
	
#ifdef EQUELLE_DEBUG
	if ( debug_vec_.size() != this->size_) {
	    std::cout << "\t\tDebug vector is of size " << debug_vec_.size() << 
		" while this->size_ is " << this->size_ << "\n";
	    std::vector<double> temp(this->size_, 0);
	    cudaStatus_ = cudaMemcpy( &temp[0], other.dev_values_,
				      sizeof(double) * this->size_,
				      cudaMemcpyDeviceToHost);
	    checkError_("cudaMemcpy(temp) in CudaArray::operator=(const CudaArray&)");
	    debug_vec_ = temp;
	}
	else {
	    cudaStatus_ = cudaMemcpy( &debug_vec_[0], other.dev_values_,
				      sizeof(double) * this->size_,
				      cudaMemcpyDeviceToHost);
	    checkError_("cudaMemcpy(debug_vec) in CudaArray::operator=(const CudaArray&)");
	}
	last_val = debug_vec_[size_-1];
#endif // EQUELLE_DEBUG


    } // if this != &other
    
    return *this;

} // Assignment copy operator!



// Destructor:
CudaArray::~CudaArray() {
    if (dev_values_ != 0) {
	cudaStatus_ = cudaFree(dev_values_);
	checkError_("cudaFree in CudaArray::~CudaArray");
    }
}

#ifdef EQUELLE_DEBUG
// Debug function to get all values to host so that they can be seen by e.g. qtcreator
void CudaArray::debug() const {
    cudaStatus_ = cudaMemcpy( &debug_vec_[0], dev_values_, sizeof(double)*size_,
			      cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy(debug_vec_) in CudaArray::debug()");
    last_val = debug_vec_[size_ - 1];
}
#endif // EQUELLE_DEBUG



const double* CudaArray::data() const {
    return dev_values_;
}

double* CudaArray::data() {
    return dev_values_;
}




kernelSetup CudaArray::setup() const {
    return setup_;
}

// Assumes that values are already allocated on host
std::vector<double> CudaArray::copyToHost() const
{
    // Fill host_vec with zeros:
    std::vector<double> host_vec(size_, 0);

    cudaStatus_ = cudaMemcpy( &host_vec[0], dev_values_, size_*sizeof(double),
			     cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy in CudaArray::copyToHost");
    
    return host_vec;
}


int CudaArray::size() const
{
    return size_;
}



void CudaArray::checkError_(const std::string& msg) const {
    if ( cudaStatus_ != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus_));
    }
}






/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- OPERATOR OVERLOADING: -----------------------------//
/////////////////////////////////////////////////////////////////////////////////



CudaArray equelleCUDA::operator-(const CudaArray& lhs, const CudaArray& rhs) {

    CudaArray out = lhs;
    kernelSetup s = out.setup();
    minus_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CudaArray equelleCUDA::operator+(const CudaArray& lhs, const CudaArray& rhs) {

    CudaArray out = lhs;
    kernelSetup s = out.setup();
    plus_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CudaArray equelleCUDA::operator*(const CudaArray& lhs, const CudaArray& rhs) {

    CudaArray out = lhs;
    kernelSetup s = out.setup();
    multiplication_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CudaArray equelleCUDA::operator/(const CudaArray& lhs, const CudaArray& rhs) {

    CudaArray out = lhs;
    kernelSetup s = out.setup();
    division_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CudaArray equelleCUDA::operator*(const Scalar lhs, const CudaArray& rhs) {
    CudaArray out = rhs;
    kernelSetup s = out.setup();
    scalMultColl_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
    return out;
}

CudaArray equelleCUDA::operator*(const CudaArray& lhs, const Scalar rhs) {
    return (rhs * lhs);
}

CudaArray equelleCUDA::operator/(const CudaArray& lhs, const Scalar rhs) {
    return ( (1/rhs) * lhs);
}

CudaArray equelleCUDA::operator/(const Scalar lhs, const CudaArray& rhs) {
    CudaArray out = rhs;
    kernelSetup s = out.setup();
    scalDivColl_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
    return out;
}

CudaArray equelleCUDA::operator-(const CudaArray& arg) {
    return -1.0*arg;
}


//  >
CollOfBool equelleCUDA::operator>(const CudaArray& lhs, const CudaArray& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>(const CudaArray& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>(const Scalar lhs, const CudaArray& rhs) {
    CollOfBool out(rhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = rhs.setup();
    comp_scalGTcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs, rhs.data(), rhs.size());
    return out;
}


// <
CollOfBool equelleCUDA::operator<(const CudaArray& lhs, const CudaArray& rhs) {
    // if   a < b   then b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const CudaArray& lhs, const Scalar rhs) {
    // if  a < b  then   b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const Scalar lhs, const CudaArray& rhs) {
    // if   a < b   then b > a
    return rhs > lhs;
}


// >=
CollOfBool equelleCUDA::operator>=(const CudaArray& lhs, const CudaArray& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGEcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>=(const CudaArray& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGEscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>=(const Scalar lhs, const CudaArray& rhs) {
    CollOfBool out(rhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = rhs.setup();
    comp_scalGEcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs, rhs.data(), rhs.size());
    return out;
}


// <= 
CollOfBool equelleCUDA::operator<=(const CudaArray& lhs, const CudaArray& rhs) {
    // if   a <= b   then b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const CudaArray& lhs, const Scalar rhs) {
    // if  a <= b  then   b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const Scalar lhs, const CudaArray& rhs) {
    // if   a <= b   then b >= a
    return rhs >= lhs;
}


// ==
CollOfBool equelleCUDA::operator==(const CudaArray& lhs, const CudaArray& rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collEQcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator==(const CudaArray& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collEQscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator==(const Scalar lhs, const CudaArray& rhs) {
    return (rhs == lhs);
}


// !=
CollOfBool equelleCUDA::operator!=(const CudaArray& lhs, const CudaArray& rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collNEcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator!=(const CudaArray& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collNEscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator!=(const Scalar lhs, const CudaArray& rhs) {
    return (rhs != lhs);
}

/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- KERNEL IMPLEMENTATIONS: ---------------------------//
/////////////////////////////////////////////////////////////////////////////////



__global__ void wrapCudaArray::minus_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] - rhs[index];
    }
}


__global__ void wrapCudaArray::plus_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if( index < size ) {
	out[index] = out[index] + rhs[index];
    }
}

__global__ void wrapCudaArray::multiplication_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] * rhs[index];
    }
}

__global__ void wrapCudaArray::division_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] / rhs[index];
    }
}

__global__ void wrapCudaArray::scalMultColl_kernel(double* out, const double scal,
						       const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index]*scal;
    }
}

__global__ void wrapCudaArray::scalDivColl_kernel(double* out, const double scal,
						     const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = scal/out[index];
    }
}
						   
__global__ void wrapCudaArray::comp_collGTcoll_kernel( bool* out,
							  const double* lhs,
							  const double* rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] > rhs[index];
    }
}

__global__ void wrapCudaArray::comp_collGTscal_kernel( bool* out,
							  const double* lhs,
							  const double rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] > rhs;
    }
}

__global__ void wrapCudaArray::comp_scalGTcoll_kernel( bool* out,
							  const double lhs,
							  const double* rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs > rhs[index];
    }
}

__global__ void wrapCudaArray::comp_collGEcoll_kernel( bool* out,
							  const double* lhs,
							  const double* rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] >= rhs[index];
    }
}

__global__ void wrapCudaArray::comp_collGEscal_kernel( bool* out,
							  const double* lhs,
							  const double rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] >= rhs;
    }
}

__global__ void wrapCudaArray::comp_scalGEcoll_kernel( bool* out,
							  const double lhs,
							  const double* rhs,
							  const int size) 
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs >= rhs[index];
    }
}


__global__ void wrapCudaArray::comp_collEQcoll_kernel( bool* out,
							  const double* lhs,
							  const double* rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = ( lhs[index] == rhs[index] );
    }
}
							
__global__ void wrapCudaArray::comp_collEQscal_kernel( bool* out,
							  const double* lhs,
							  const double rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = ( lhs[index] == rhs );
    }
}
							

__global__ void wrapCudaArray::comp_collNEcoll_kernel( bool* out,
							  const double* lhs,
							  const double* rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = ( lhs[index] != rhs[index] );
    }
}
							
__global__ void wrapCudaArray::comp_collNEscal_kernel( bool* out,
							  const double* lhs,
							  const double rhs,
							  const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = ( lhs[index] != rhs );
    }
}




// Transforming CollOfBool
std::vector<bool> equelleCUDA::cob_to_std( const CollOfBool& cob) {
    thrust::host_vector<bool> host = cob;
    return std::vector<bool>(host.begin(), host.end());
}