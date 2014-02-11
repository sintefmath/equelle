#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>

//#include <string>
//#include <fstream>
//#include <iterator>
//#include <cuda.h>

#include <stdlib.h>

//#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_cuda.hpp"

// Implementation of the class CollOfScalar

CollOfScalar::CollOfScalar()
{
    // Blank
    values = 0;
    size = 0;

}

CollOfScalar::CollOfScalar(int size) {
    // dev_vec.reserve(size);
    this->size = size;
    values = (double*)malloc(size*sizeof(double));
}

// Destructor:
CollOfScalar::~CollOfScalar() {
    if ( size > 0 ) {
	free(values);
	size = 0;
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
