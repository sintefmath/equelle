#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

#include <string>
//#include <fstream>
//#include <iterator>
#include <cuda.h>

#include <stdlib.h>

#include "EquelleRuntimeCUDA_cuda.hpp"

// Implementation of the class CollOfScalar

//CollOfScalar::CollOfScalar()
//{
//   //values = 0;
//   size = 0;
//  dev_values = 0;
//  //dev_vec = thrust::device_vector<double>(0);
//
//}

CollOfScalar::CollOfScalar(int size) : size(size), dev_values(0) {
    cudaStatus = cudaMalloc( (void**)&dev_values, size*sizeof(double));
    checkError("cudaMalloc in CollOfScalar::CollOfScalar(int)");

    // Set grid and block size for cuda kernel executions:
    block_x = havahol_helper::MAX_THREADS;
    grid_x = (size + block_x - 1) / block_x;

}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) : size(0), dev_values(0) {
    std::cout << "Copy constructor!\n";
    size = coll.size;
   
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

double* CollOfScalar::getDevValues() const {
    return dev_values;
}

int CollOfScalar::block() {
    return block_x;
}

int CollOfScalar::grid() {
    return grid_x;
}

// Assumes that values are already allocated on host
void CollOfScalar::copyToHost(double* values) const
{
    std::cout << "copyToHost() - dev_values = " << dev_values << std::endl;
    
    cudaStatus = cudaMemcpy( values, dev_values, size*sizeof(double),
					cudaMemcpyDeviceToHost);
    checkError("cudaMemcpy in CollOfScalar::copyToHost");
}


void CollOfScalar::setValuesFromFile(std::istream_iterator<double> begin,
				     std::istream_iterator<double> end)
{
    thrust::host_vector<double> host_vec(begin, end);

    double* values = (double*)malloc(size*sizeof(double));
    for(int i = 0; i < host_vec.size(); i++) {
	values[i] = host_vec[i];
    }
    //dev_vec = host_vec;
    cudaStatus = cudaMemcpy( dev_values, values, size*sizeof(double),
					 cudaMemcpyHostToDevice);
    checkError("cudamMemcpy in CollOfScalar::setValuesFromFile");
    
    free(values);
}

void CollOfScalar::setValuesUniform(double val)
{
    // Can not use cudaMemset as it sets float values on a given
    // number of bytes.
    double* host = (double*)malloc(size*sizeof(double));
    for (int i = 0; i < size; ++i) {
	host[i] = val;
    }
    
    cudaStatus = cudaMemcpy(dev_values, host, size*sizeof(double),
				    cudaMemcpyHostToDevice);
    checkError("cudaMemcpy in CollOfScalar::setValuesUniform");
    
    free(host);
}

int CollOfScalar::getSize() const
{
    //return dev_vec.size();
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
    //double* lhs_dev = lhs.getDevValues();
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();

    dim3 block(out.block());
    dim3 grid(out.grid());
    std::cout << "Calling minus_kernel!\n";
    minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();

    dim3 block(out.block());
    dim3 grid(out.grid());
    plus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();

    dim3 block(out.block());
    dim3 grid(out.grid());
    multiplication_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    return out;
}

CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();

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


