//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/fill.h>
//#include <thrust/copy.h>
//#include <thrust/sequence.h>

#include <string>
//#include <fstream>
//#include <iterator>
#include <cuda.h>

#include <stdlib.h>
#include <vector>
#include <iostream>

// For error exception macro:
#include <opm/core/utility/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA_cuda.hpp"

// Implementation of the class CollOfScalar

CollOfScalar::CollOfScalar() 
    : size_(0), 
      dev_values(0),
      block_x_(0),
      grid_x_(0)
{
}

CollOfScalar::CollOfScalar(const int size) 
    : size_(size),
      dev_values(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    cudaStatus = cudaMalloc( (void**)&dev_values, size_*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(int)");
}

CollOfScalar::CollOfScalar(const int size, const int value) 
    : size_(size),
      dev_values(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.

    std::vector<double> host_vec(size_, value);

    cudaStatus = cudaMalloc( (void**)&dev_values, size_*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(int, int)");
        
    cudaStatus = cudaMemcpy(dev_values, &host_vec[0], size_*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError("cudaMemcpy in CollOfScalar::CollOfScalar(int, int)");
} 


// Constructor from vector, in order to do testing
CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : size_(host_vec.size()),
      dev_values(0),
      block_x_(equelleCUDA::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    cudaStatus = cudaMalloc( (void**)&dev_values, size_*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(std::vector<double>)");
    
    cudaStatus = cudaMemcpy(dev_values, &host_vec[0], size_*sizeof(double),
			    cudaMemcpyHostToDevice);
    checkError("cudaMemcpy in CollOfScalar::CollOfScalar(std::vector<double>)");
}

// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) 
    : size_(coll.size_), 
      dev_values(0),
      grid_x_(coll.grid_x_),
      block_x_(coll.block_x_)
{
    std::cout << "Copy constructor!\n";
   
    if (coll.dev_values != 0) {
	cudaStatus = cudaMalloc( (void**)&dev_values, size_*sizeof(double));
	checkError("cudaMalloc in CollOfScalar::CollOfScalar(const CollOfScalar&)"); 

	cudaStatus = cudaMemcpy(dev_values, coll.dev_values, size_*sizeof(double),
			    cudaMemcpyDeviceToDevice);
	checkError("cudaMemcpy in CollOfScalar::CollOfScalar(const CollOfScalar&)");
    }    

}


// Destructor:
CollOfScalar::~CollOfScalar() {
    if (dev_values != 0) {
	cudaStatus = cudaFree(dev_values);
	checkError("cudaFree in CollOfScalar::~CollOfScalar");
    }
}

const double* CollOfScalar::data() const {
    return dev_values;
}

double* CollOfScalar::data() {
    return dev_values;
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
    
    std::vector<double> host_vec(size_, 0);

    cudaStatus = cudaMemcpy( &host_vec[0], dev_values, size_*sizeof(double),
			     cudaMemcpyDeviceToHost);
    checkError("cudaMemcpy in CollOfScalar::copyToHost");
    
    return host_vec;
}


int CollOfScalar::size() const
{
    return size_;
}


void CollOfScalar::checkError(const std::string& msg) const {
    if ( cudaStatus != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus));
	//std::cout <<  "Cuda error\n\t" << msg << "\n\tError code: " << cudaGetErrorString(cudaStatus) << std::endl;
	//exit(0);
    }
}




/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- OPERATOR OVERLOADING: -----------------------------//
/////////////////////////////////////////////////////////////////////////////////



CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    //double* lhs_dev = lhs.data();
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    std::cout << "Calling minus_kernel!\n";
    equelleCUDA::minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    equelleCUDA::plus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    equelleCUDA::multiplication_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    const double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    equelleCUDA::division_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
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


