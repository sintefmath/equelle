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

#include "CollOfScalar.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"


// Implementation of the class CollOfScalar

using namespace equelleCUDA;


CollOfScalar::CollOfScalar() 
    : size_(0), 
      dev_values_(0),
      block_x_(0),
      grid_x_(0)
{
    // Intentionally left blank
}

CollOfScalar::CollOfScalar(const int size) 
    : size_(size),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CollOfScalar::CollOfScalar(int)");
}

CollOfScalar::CollOfScalar(const int size, const double value) 
    : size_(size),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.

    std::vector<double> host_vec(size_, value);

    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CollOfScalar::CollOfScalar(int, int)");
        
    cudaStatus_ = cudaMemcpy(dev_values_, &host_vec[0], size_*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy in CollOfScalar::CollOfScalar(int, int)");
} 


// Constructor from vector, in order to do testing
CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : size_(host_vec.size()),
      dev_values_(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CollOfScalar::CollOfScalar(std::vector<double>)");
    
    cudaStatus_ = cudaMemcpy(dev_values_, &host_vec[0], size_*sizeof(double),
			    cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy in CollOfScalar::CollOfScalar(std::vector<double>)");
}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) 
    : size_(coll.size_), 
      dev_values_(0),
      grid_x_(coll.grid_x_),
      block_x_(coll.block_x_)
{
    std::cout << "Copy constructor!\n";
   
    if (coll.dev_values_ != 0) {
	cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
	checkError_("cudaMalloc in CollOfScalar::CollOfScalar(const CollOfScalar&)"); 

	cudaStatus_ = cudaMemcpy(dev_values_, coll.dev_values_, size_*sizeof(double),
				 cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy in CollOfScalar::CollOfScalar(const CollOfScalar&)");
    }
    
}


// Copy assignment operator
CollOfScalar& CollOfScalar::operator= (const CollOfScalar& other) {
    
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
	
	std::cout << "COPY ASSIGNMENT OPERATOR ( this->size = " << this->size_ << ")\n";

	if ( this->size_ != other.size_) {

	    std::cout << "\tCHANGING SIZE FROM " << this->size_ << " TO " << other.size_ << "\n";
		
	    // If different size: Is this even allowed?
	    // Free memory:
	    cudaStatus_ = cudaFree(this->dev_values_);
	    checkError_("cudaFree(this->dev_values_) in CollOfScalar::operator=(const CollOfScalar&)");
	    // Allocate new memory:
	    cudaStatus_ = cudaMalloc((void**)&this->dev_values_,
				     sizeof(double) * other.size_);
	    checkError_("cudaMalloc(this->dev_values_) in CollOfScalar::operator=(const CollOfScalar&)");
	    
	    // Set variables depending on size_:
	    this->size_ = other.size_;
	    this->block_x_ = other.block_x_;
	    this->grid_x_ = other.grid_x_;
	}

	// Copy memory block from other to this:
	cudaStatus_ = cudaMemcpy( this->dev_values_, other.dev_values_,
				  sizeof(double) * this->size_,
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(dev_values_) in CollOfScalar::operator=(const CollOfScalar)");
    } // if this == &other
    
    return *this;

}


// Destructor:
CollOfScalar::~CollOfScalar() {
    if (dev_values_ != 0) {
	cudaStatus_ = cudaFree(dev_values_);
	checkError_("cudaFree in CollOfScalar::~CollOfScalar");
    }
}

const double* CollOfScalar::data() const {
    return dev_values_;
}

double* CollOfScalar::data() {
    return dev_values_;
}



int CollOfScalar::block() const {
    return block_x_;
}

int CollOfScalar::grid() const {
    return grid_x_;
}

// Assumes that values are already allocated on host
std::vector<double> CollOfScalar::copyToHost() const
{
    //std::cout << "copyToHost() - val_ptr = " << dev_values << std::endl;
    
    // Fill host_vec with zeros:
    std::vector<double> host_vec(size_, 0);

    cudaStatus_ = cudaMemcpy( &host_vec[0], dev_values_, size_*sizeof(double),
			     cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy in CollOfScalar::copyToHost");
    
    return host_vec;
}


int CollOfScalar::size() const
{
    return size_;
}



void CollOfScalar::checkError_(const std::string& msg) const {
    if ( cudaStatus_ != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus_));
	//std::cout <<  "Cuda error\n\t" << msg << "\n\tError code: " << cudaGetErrorString(cudaStatus) << std::endl;
	//exit(0);
    }
}






/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- OPERATOR OVERLOADING: -----------------------------//
/////////////////////////////////////////////////////////////////////////////////



CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    //double* lhs_dev = lhs.data();
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    std::cout << "Calling minus_kernel!\n";
    minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    plus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    multiplication_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    division_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const Scalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out = rhs;
    dim3 block(out.block());
    dim3 grid(out.grid());
    multScalCollection_kernel<<<grid,block>>>(out.data(), lhs, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const Scalar& rhs) {
    return (rhs * lhs);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const Scalar& rhs) {
    return ( (1/rhs) * lhs);
}

CollOfScalar equelleCUDA::operator/(const Scalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out = rhs;
    dim3 block(out.block());
    dim3 grid(out.grid());
    divScalCollection_kernel<<<grid,block>>>(out.data(), lhs, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg) {
    return -1.0*arg;
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    dim3 block(lhs.block());
    dim3 grid(lhs.grid());
    compGTkernel<<<grid,block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- KERNEL IMPLEMENTATIONS: ---------------------------//
/////////////////////////////////////////////////////////////////////////////////



__global__ void equelleCUDA::minus_kernel(double* out, const double* rhs, const int size) {
    
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] - rhs[index];
    }
}


__global__ void equelleCUDA::plus_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if( index < size ) {
	out[index] = out[index] + rhs[index];
    }
}

__global__ void equelleCUDA::multiplication_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] * rhs[index];
    }
}

__global__ void equelleCUDA::division_kernel(double* out, const double* rhs, const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] / rhs[index];
    }
}

__global__ void equelleCUDA::multScalCollection_kernel(double* out, const double scal,
						       const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index]*scal;
    }
}

__global__ void equelleCUDA::divScalCollection_kernel(double* out, const double scal,
						     const int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = scal/out[index];
    }
}
						   
__global__ void equelleCUDA::compGTkernel( bool* out,
					   const double* lhs,
					   const double* rhs,
					   const int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] > rhs[index];
    }
}


// Transforming CollOfBool
std::vector<bool> equelleCUDA::cob_to_std( const CollOfBool& cob) {
    thrust::host_vector<bool> host = cob;
    return std::vector<bool>(host.begin(), host.end());
}