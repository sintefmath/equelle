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

#include "EquelleRuntimeCUDA_cuda.hpp"

// Implementation of the class CollOfScalar

CollOfScalar::CollOfScalar() : size(0), dev_values(0)
{
}

CollOfScalar::CollOfScalar(int size) : size(size), dev_values(0) {
    cudaStatus = cudaMalloc( (void**)&dev_values, size*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(int)");

    // Set grid and block size for cuda kernel executions:
    block_x = havahol_helper::MAX_THREADS;
    grid_x = (size + block_x - 1) / block_x;

}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) : size(coll.size), dev_values(0) {
    std::cout << "Copy constructor!\n";
   
    if (coll.dev_values != 0) {
	cudaStatus = cudaMalloc( (void**)&dev_values, size*sizeof(double));
	checkError("cudaMalloc in CollOfScalar::CollOfScalar(const CollOfScalar&)"); 

	cudaStatus = cudaMemcpy(dev_values, coll.dev_values, size*sizeof(double),
			    cudaMemcpyDeviceToDevice);
	checkError("cudaMemcpy in CollOfScalar::CollOfScalar(const CollOfScalar&)");
    }    
    grid_x = coll.grid_x;
    block_x = coll.block_x;
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

int CollOfScalar::block() {
    return block_x;
}

int CollOfScalar::grid() {
    return grid_x;
}

// Assumes that values are already allocated on host
std::vector<double> CollOfScalar::copyToHost() const
{
    std::cout << "copyToHost() - val_ptr = " << dev_values << std::endl;
    
    std::vector<double> host_vec;
    host_vec.reserve(size);

    cudaStatus = cudaMemcpy( &host_vec[0], dev_values, size*sizeof(double),
					cudaMemcpyDeviceToHost);
    checkError("cudaMemcpy in CollOfScalar::copyToHost");
    
    return host_vec;
}


void CollOfScalar::setValuesFromFile(std::istream_iterator<double> begin,
				     std::istream_iterator<double> end)
{
    std::vector<double> host_vec(begin, end);
    double* values = &host_vec[0];

    cudaStatus = cudaMemcpy( dev_values, values, size*sizeof(double),
					 cudaMemcpyHostToDevice);
    checkError("cudamMemcpy in CollOfScalar::setValuesFromFile");    
}

void CollOfScalar::setValuesUniform(double val)
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.

    std::vector<double> host_vec(size, val);
        
    cudaStatus = cudaMemcpy(dev_values, &host_vec[0], size*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError("cudaMemcpy in CollOfScalar::setValuesUniform");
} 
    
int CollOfScalar::getSize() const
{
    return size;
}



// TODO: Replace exit(0) with a smoother and more correct exit strategy.
void CollOfScalar::checkError(const std::string& msg) const {
    if ( cudaStatus != cudaSuccess ) {
	//OPM_THROW(std::runtime_error, "Cuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus));
	// OPM_THROW does not work as we cannot include OPM in this cuda file.
	std::cout <<  "Cuda error\n\t" << msg << "\n\tError code: " << cudaGetErrorString(cudaStatus) << std::endl;
	exit(0);
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
    minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    plus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    multiplication_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.data();
    double* out_dev = out.data();

    dim3 block(out.block());
    dim3 grid(out.grid());
    division_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
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


