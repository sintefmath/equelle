#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

//#include <string>
//#include <fstream>
//#include <iterator>
#include <cuda.h>

#include <stdlib.h>

//#include "EquelleRuntimeCUDA.hpp"
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

CollOfScalar::CollOfScalar(int size) {
    // dev_vec.reserve(size);
    this->size = size;
    //values = (double*)malloc(size*sizeof(double));
    //dev_vec = thrust::device_vector<double>(size);
    cudaError_t status = cudaMalloc( (void**)&dev_values, size*sizeof(double));
    if ( status != cudaSuccess ) {
	std::cout << "Error allocating dev_values in CollOfScalar(int)\n";
	exit(0);
    }

    // Set grid and block size for cuda kernel executions:
    block_x = havahol_helper::MAX_THREADS;
    grid_x = (size + block_x - 1) / block_x;

}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) {
    std::cout << "Copy constructor!\n";
    size = coll.size;
    //values = 0;
    dev_values = 0;
    //if (coll.values != 0) {
    //	values = (double*)malloc(size*sizeof(double));
    //	for ( int i = 0; i < size; i++) {
    //	    values[i] = coll.values[i];
    //	}
    //}
    if (coll.dev_values != 0) {
	cudaError_t status = cudaMalloc( (void**)&dev_values, size*sizeof(double));
	if ( status != cudaSuccess ) {
	    std::cout << "Error allocating dev_values in CollOfScalar(CollOfScalar)\n";
	    exit(0);
	}
	status = cudaMemcpy(dev_values, coll.dev_values, size*sizeof(double),
			    cudaMemcpyDeviceToDevice);
	if ( status != cudaSuccess ){
	    std::cout << "Error copying dev_values in copy constructor\n";
	    exit(0);
	}
    }    
    grid_x = coll.grid_x;
    block_x = coll.block_x;
}


// Destructor:
CollOfScalar::~CollOfScalar() {
    if ( size > 0 ) {
	size = 0;
    }
    //if (values != 0) {
    //	std::cout << "Freeing values\n";
    //	free(values);
    //	//values = 0;
    //}
    if (dev_values != 0) {
	cudaError_t status = cudaFree(dev_values);
	if (status != cudaSuccess) {
	    std::cout << "Error cuda-freeing in destructor of CollOfScalar\n";
	    std::cout << "\tError code: " << cudaGetErrorString(status) << std::endl;
	    exit(0);
	}
	//dev_values = 0;
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
    
    cudaError_t cudaError = cudaMemcpy( values, dev_values, size*sizeof(double),
					cudaMemcpyDeviceToHost);
    if (cudaError != cudaSuccess) {
	std::cout << "Error copying to host in output. \n\tError code = ";
	std::cout << cudaGetErrorString(cudaError) << "\n";
	exit(0);
    }
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
    cudaError_t cudaStatus = cudaMemcpy( dev_values, values, size*sizeof(double),
					 cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
	std::cout << "Error in cudaMemcpy to dev from file.\n";
	exit(0);
    }
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
    
    cudaError_t status = cudaMemcpy(dev_values, host, size*sizeof(double),
				    cudaMemcpyHostToDevice);
    if ( status != cudaSuccess ) {
	std::cout << "Error in uniform value initialization\n";
	std::cout << "\tError code: " << cudaGetErrorString(status) << std::endl;
	exit(0);
    }
    free(host);
}

int CollOfScalar::getSize() const
{
    //return dev_vec.size();
    return size;
}


/// OPERATION OVERLOADING
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

/// KERNEL IMPLEMENTATIONS:
__global__ void minus_kernel(double* out, double* rhs, int size) {
    
    int index = threadIdx.x + blockDim.x*blockIdx.x;
    if ( index < size) {
	out[index] = out[index] - rhs[index];
    }
}