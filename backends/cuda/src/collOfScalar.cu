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
      grid_x_(0),
      setup_(0)
#ifdef EQUELLE_DEBUG
    , debug_vec_(0)
#endif // EQUELLE_DEBUG
{
    // Intentionally left blank
}

// Allocating memory without initialization
CollOfScalar::CollOfScalar(const int size) 
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
    checkError_("cudaMalloc in CollOfScalar::CollOfScalar(int)");
#ifdef EQUELLE_DEBUG
    std::cout << "Debug mode is on!\n";
#endif // EQUELLE_DEBUG
}

CollOfScalar::CollOfScalar(const int size, const double value) 
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
    std::cerr << __PRETTY_FUNCTION__ << std::endl;

    std::vector<double> host_vec(size_, value);

    cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
    checkError_("cudaMalloc in CollOfScalar::CollOfScalar(int, double)");
        
    cudaStatus_ = cudaMemcpy(dev_values_, &host_vec[0], size_*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy in CollOfScalar::CollOfScalar(int, double)");

} 


// Constructor from vector, in order to do testing
CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
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
      block_x_(coll.block_x_),
      setup_(size_)
#ifdef EQUELLE_DEBUG
    , debug_vec_(coll.size_, 0)
#endif // EQUELLE_DEBUG
{
    std::cout << "Copy constructor!\n";
    std::cerr << __PRETTY_FUNCTION__ << std::endl;    

    if (coll.dev_values_ != 0) {
	cudaStatus_ = cudaMalloc( (void**)&dev_values_, size_*sizeof(double));
	checkError_("cudaMalloc in CollOfScalar::CollOfScalar(const CollOfScalar&)"); 

	cudaStatus_ = cudaMemcpy(dev_values_, coll.dev_values_, size_*sizeof(double),
				 cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy in CollOfScalar::CollOfScalar(const CollOfScalar&)");
    }
    
#ifdef EQUELLE_DEBUG
    // Copy value to the std::vector debug_vec_
    std::cout << "\tDEBUG IS ON!\n";
    if (coll.dev_values_ != 0 ) {
	cudaStatus_ = cudaMemcpy( &debug_vec_[0], dev_values_, size_*sizeof(double),
				  cudaMemcpyDeviceToHost );
	checkError_("cudaMemcpy for DEBUG in CollOfScalar::CollOfScalar(const CollOfScalar&)");
	last_val = debug_vec_[size_ - 1];
    }
#endif // EQUELLE_DEBUG
}


// Copy assignment operator
CollOfScalar& CollOfScalar::operator= (const CollOfScalar& other) {
    std::cerr << __PRETTY_FUNCTION__ << std::endl;    

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
	checkError_("cudaMemcpy(dev_values_) in CollOfScalar::operator=(const CollOfScalar&)");
	
#ifdef EQUELLE_DEBUG
	if ( debug_vec_.size() != this->size_) {
	    std::cout << "\t\tDebug vector is of size " << debug_vec_.size() << 
		" while this->size_ is " << this->size_ << "\n";
	    std::vector<double> temp(this->size_, 0);
	    cudaStatus_ = cudaMemcpy( &temp[0], other.dev_values_,
				      sizeof(double) * this->size_,
				      cudaMemcpyDeviceToHost);
	    checkError_("cudaMemcpy(temp) in CollOfScalar::operator=(const CollOfScalar&)");
	    debug_vec_ = temp;
	}
	else {
	    cudaStatus_ = cudaMemcpy( &debug_vec_[0], other.dev_values_,
				      sizeof(double) * this->size_,
				      cudaMemcpyDeviceToHost);
	    checkError_("cudaMemcpy(debug_vec) in CollOfScalar::operator=(const CollOfScalar&)");
	}
	last_val = debug_vec_[size_-1];
#endif // EQUELLE_DEBUG


    } // if this != &other
    
    return *this;

} // Assignment copy operator!



// Destructor:
CollOfScalar::~CollOfScalar() {
    if (dev_values_ != 0) {
	cudaStatus_ = cudaFree(dev_values_);
	checkError_("cudaFree in CollOfScalar::~CollOfScalar");
    }
}

#ifdef EQUELLE_DEBUG
// Debug function to get all values to host so that they can be seen by e.g. qtcreator
void CollOfScalar::debug() const {
    cudaStatus_ = cudaMemcpy( &debug_vec_[0], dev_values_, sizeof(double)*size_,
			      cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy(debug_vec_) in CollOfScalar::debug()");
    last_val = debug_vec_[size_ - 1];
}
#endif // EQUELLE_DEBUG



const double* CollOfScalar::data() const {
    return dev_values_;
}

double* CollOfScalar::data() {
    return dev_values_;
}




kernelSetup CollOfScalar::setup() const {
    return setup_;
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

    std::cout << "Calling minus_kernel!\n";
    kernelSetup s = out.setup();
    minus_kernel <<<s.grid, s.block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    kernelSetup s = out.setup();
    plus_kernel <<<s.grid, s.block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    kernelSetup s = out.setup();
    multiplication_kernel <<<s.grid, s.block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    kernelSetup s = out.setup();
    division_kernel <<<s.grid, s.block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const Scalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out = rhs;
    kernelSetup s = out.setup();
    multScalCollection_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
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
    kernelSetup s = out.setup();
    divScalCollection_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg) {
    return -1.0*arg;
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collLTcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collLTscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
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
						   
__global__ void equelleCUDA::comp_collGTcoll_kernel( bool* out,
						     const double* lhs,
						     const double* rhs,
						     const int size)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] > rhs[index];
    }
}

__global__ void equelleCUDA::comp_collGTscal_kernel( bool* out,
						     const double* lhs,
						     const double rhs,
						     const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] > rhs;
    }
}

__global__ void equelleCUDA::comp_collLTcoll_kernel( bool* out,
						     const double* lhs,
						     const double* rhs,
						     const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] < rhs[index];
    }
}

__global__ void equelleCUDA::comp_collLTscal_kernel( bool* out,
						     const double* lhs,
						     const double rhs,
						     const int size)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	out[index] = lhs[index] < rhs;
    }
}




// Transforming CollOfBool
std::vector<bool> equelleCUDA::cob_to_std( const CollOfBool& cob) {
    thrust::host_vector<bool> host = cob;
    return std::vector<bool>(host.begin(), host.end());
}