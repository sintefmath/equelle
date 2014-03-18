
#include <cuda.h>
#include <cuda_runtime.h>

#include <opm/core/utility/ErrorMacros.hpp>
#include <iostream>
#include <vector>
#include <math.h>



#include "CollOfVector.hpp"
#include "CollOfScalar.hpp"


using namespace equelleCUDA;


CollOfVector::CollOfVector() 
    : CollOfScalar()
{
    // intentionally left blank
}

CollOfVector::CollOfVector(const int size, const int dim)
    : CollOfScalar(size*dim), dim_(dim)
{
    // intentionally left blank
}

CollOfVector::CollOfVector(const std::vector<double>& host, const int dim)
    : CollOfScalar(host), dim_(dim)
{
    // intentionally left blank
}


// Copy assignment operator
CollOfVector& CollOfVector::operator= (const CollOfVector& other) {
    // Call the copy assignment operator for the base class:
    CollOfScalar::operator=(other);
    this->dim_ = other.dim();
    return *this;
}

// Copy-constructor
CollOfVector::CollOfVector(const CollOfVector& coll)
    : CollOfScalar(coll), dim_(coll.dim_)
{
    // intentionally left blank
}
  


// Destructor
CollOfVector::~CollOfVector()
{
    // intentionally left blank.
}

//  ----- NORM -----
CollOfScalar CollOfVector::norm() const {
    CollOfScalar out(numVectors());
    dim3 block(out.block());
    dim3 grid(out.grid());
    normKernel<<<grid, block>>>(out.data(), data(), numVectors(), dim());
    return out;
}





int CollOfVector::dim() const {
    if ( dim_ == 0 ) {
	OPM_THROW(std::runtime_error, "Calling numVectors() on a CollOfVector of dimension 0\n --> Dividing by zero!");
    }
    return dim_;
}

int CollOfVector::numVectors() const {
    return size()/dim_;
}

//Operator []

CollOfScalar CollOfVector::operator[](const int index) const {
    
    if ( index < 0 || index >= dim_) {
	OPM_THROW(std::runtime_error, "Illigal dimension index " << index << " for a vector of dimension " << dim_);
    }
    
    CollOfScalar out(numVectors());
    
    dim3 block(out.block());
    dim3 grid(out.grid());

    collOfVectorOperatorIndexKernel<<<grid,block>>>( out.data(),
						     this->data(),
						     out.size(),
						     index,
						     dim_);
						    
    return out;
}

__global__ void equelleCUDA::collOfVectorOperatorIndexKernel( double* out,
							      const double* vec,
							      const int size_out,
							      const int index,
							      const int dim)
{
    // Index:
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if ( i < size_out ) {
	out[i] = vec[i*dim + index];
    }
}



// --------- NORM KERNEL ---------------------

__global__ void equelleCUDA::normKernel( double* out,
					 const double* vectors,
					 const int numVectors,
					 const int dim)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < numVectors ){
	double norm = 0;
	for ( int i = 0; i < dim; i++) {
	    norm += vectors[index*dim + i]*vectors[index*dim + i];
	}
	out[index] = sqrt(norm);
    }
}



// ------------- OPERATOR OVERLOADING --------------------

CollOfVector equelleCUDA::operator+(const CollOfVector& lhs, const CollOfVector& rhs) {

    CollOfVector out = lhs;
    dim3 block(out.block());
    dim3 grid(out.grid());
    plus_kernel<<<grid,block>>>(out.data(), rhs.data(), out.size());
    return out;
}


CollOfVector equelleCUDA::operator-(const CollOfVector& lhs, const CollOfVector& rhs) {
    CollOfVector out = lhs;
    dim3 block(out.block());
    dim3 grid(out.grid());
    minus_kernel<<<grid,block>>>(out.data(), rhs.data(), out.size());
    return out;
}