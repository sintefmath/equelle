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

CollOfScalar::CollOfScalar()
{
    // Blank
    values = 0;
    size = 0;
    dev_values = 0;
    //dev_vec = thrust::device_vector<double>(0);

}

CollOfScalar::CollOfScalar(int size) {
    // dev_vec.reserve(size);
    this->size = size;
    values = (double*)malloc(size*sizeof(double));
    //dev_vec = thrust::device_vector<double>(size);
    cudaError_t status = cudaMalloc( (void**)&dev_values, size*sizeof(double));
    if ( status != cudaSuccess ) {
	std::cout << "Error allocating dev_values in CollOfScalar(int)\n";
	exit(0);
    }
}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll) {
    std::cout << "Copy constructor!\n";
    size = coll.size;
    values = 0;
    dev_values = 0;
    if (coll.values != 0) {
	values = (double*)malloc(size*sizeof(double));
	for ( int i = 0; i < size; i++) {
	    values[i] = coll.values[i];
	}
    }
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
}


// Destructor:
CollOfScalar::~CollOfScalar() {
    if ( size > 0 ) {
	size = 0;
    }
    if (values != 0) {
	std::cout << "Freeing values\n";
	free(values);
	//values = 0;
    }
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

//double* CollOfScalar::getHostValues() const {
//    return values;
//}

void CollOfScalar::copyToHost() const
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

double CollOfScalar::getValue(int index) const
{
    if ( index > -1 && index < size) {
	return values[index];
    }
    else {
	exit(0);
	return 0;
    }
}

void CollOfScalar::setValue(int index, double value)
{
    if (index > -1 && index < size) {
	values[index] = value;
    }
}

void CollOfScalar::setValuesFromFile(std::istream_iterator<double> begin,
				     std::istream_iterator<double> end)
{
    thrust::host_vector<double> host_vec(begin, end);
    //for( std::istream_iterator<double> i = begin; i != end; i++) {
    //dev_vec.insert(dev_vec.begin(), begin, end);
    //}
    //double* a;
    //cudaError_t t = cudaMalloc( (void**)&a, sizeof(double)*dev_vec.size());
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
}

void CollOfScalar::setValuesUniform(double val, int size)
{
    //dev_vec.reserve(size);
    //thrust::fill(dev_vec.begin(), dev_vec.begin() + 4, val);
    // dev_vec.insert(dev_vec.begin(), dev_vec.begin() + size, val);
       //dev_vec.push_back(1.0);
}

int CollOfScalar::getSize() const
{
    //return dev_vec.size();
    return size;
}
