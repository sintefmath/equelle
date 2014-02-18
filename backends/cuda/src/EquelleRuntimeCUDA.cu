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

CollOfScalar::CollOfScalar(int size) 
    : size_(size),
      dev_values(0),
      block_x_(havahol_helper::MAX_THREADS),
      grid_x_((size_ + block_x_ - 1) / block_x_)
{
    cudaStatus = cudaMalloc( (void**)&dev_values, size_*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(int)");
}

// Constructor from vector, in order to do testing
CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : size_(host_vec.size()),
      dev_values(0),
      block_x_(havahol_helper::MAX_THREADS),
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

double* CollOfScalar::data() const {
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
    
    // Fill the vector with zeros. That way host_vec.size() behaves as expected.
    std::vector<double> host_vec(size_, 0);

    cudaStatus = cudaMemcpy( &host_vec[0], dev_values, size_*sizeof(double),
					cudaMemcpyDeviceToHost);
    checkError("cudaMemcpy in CollOfScalar::copyToHost");
    
    return host_vec;
}


void CollOfScalar::setValuesFromFile(std::istream_iterator<double> begin,
				     std::istream_iterator<double> end)
{
    std::vector<double> host_vec(begin, end);
    double* values = &host_vec[0];

    if ( host_vec.size() != size_ ) {
	OPM_THROW(std::runtime_error, "wrong size of input file collection");
    }

    cudaStatus = cudaMemcpy( dev_values, values, size_*sizeof(double),
					 cudaMemcpyHostToDevice);
    checkError("cudamMemcpy in CollOfScalar::setValuesFromFile");    
}

void CollOfScalar::setValuesUniform(double val)
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.

    std::vector<double> host_vec(size_, val);
        
    cudaStatus = cudaMemcpy(dev_values, &host_vec[0], size_*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError("cudaMemcpy in CollOfScalar::setValuesUniform");
} 
    
int CollOfScalar::size() const
{
    return size_;
}



// TODO: Replace exit(0) with a smoother and more correct exit strategy.
void CollOfScalar::checkError(const std::string& msg) const {
    if ( cudaStatus != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus));
	// OPM_THROW does not work as we cannot include OPM in this cuda file.
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
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    std::cout << "Calling minus_kernel!\n";
    minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    plus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    multiplication_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}

CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    division_kernel <<<grid, block>>>(out_dev, rhs_dev, out.size());
    return out;
}






/////////////////////////////////////////////////////////////////////////////////
/// ----------------------- KERNEL IMPLEMENTATIONS: ---------------------------//
/////////////////////////////////////////////////////////////////////////////////



__global__ void minus_kernel(double* out, double* rhs, int size) {
    
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size) {
	out[index] = out[index] - rhs[index];
    }
}


__global__ void plus_kernel(double* out, double* rhs, int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if( index < size ) {
	out[index] = out[index] + rhs[index];
    }
}

__global__ void multiplication_kernel(double* out, double* rhs, int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] * rhs[index];
    }
}

__global__ void division_kernel(double* out, double* rhs, int size) {
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size ) {
	out[index] = out[index] / rhs[index];
    }
}


