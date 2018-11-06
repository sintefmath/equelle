
#include <cuda.h>
#include <cuda_runtime.h>

#include <opm/common/ErrorMacros.hpp>
#include <iostream>
#include <vector>
#include <math.h>



#include "CollOfVector.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"
#include "device_functions.cuh"

using namespace equelleCUDA;
using namespace wrapCollOfVector;

CollOfVector::CollOfVector() 
    : elements_(),
      dim_(1)
{
    // intentionally left blank
}



CollOfVector::CollOfVector(const int size, const int dim)
    : elements_(size*dim), 
      dim_(dim)
{
    // intentionally left blank
}

CollOfVector::CollOfVector(const std::vector<double>& host, const int dim)
    : elements_(host), 
      dim_(dim)
{
    // intentionally left blank
}


// Copy assignment operator
CollOfVector& CollOfVector::operator= (const CollOfVector& other) {

    // Does not give sense to assign Vectors of different dimensions.
    if ( this->dim_ != other.dim_ ) {
	OPM_THROW(std::runtime_error, "Trying to assign a vector with another vector of different dim. lhs.dim_ = " << this->dim_ << " and rhs.dim_ = " << other.dim_);
    }
    this->elements_ = other.elements_;
    return *this;
}

// Move assignment operator
CollOfVector& CollOfVector::operator=(CollOfVector&& other) {
    if ( this->dim_ != other.dim_ ) {
        OPM_THROW(std::runtime_error, "Trying to assign a vector with another vector of different dim. lhs.dim_ = " << this->dim_ << " and rhs.dim_ = " << other.dim_);
    }
    this->elements_ = std::move(other.elements_);
    return *this;
}

// Copy-constructor
CollOfVector::CollOfVector(const CollOfVector& coll)
    : elements_(coll.elements_), 
      dim_(coll.dim_)
{
    // intentionally left blank
}
  
// Move constructor
CollOfVector::CollOfVector(CollOfVector&& coll)
    : elements_(std::move(coll.elements_)), 
      dim_(coll.dim_)
{
}

// Destructor
CollOfVector::~CollOfVector()
{
    // intentionally left blank.
}

// ------------ MEMBER FUNCTIONS -------------------- // 


//  ----- NORM -----
CollOfScalar CollOfVector::norm() const {
    CollOfScalar out(numVectors());
    // One thread for each vector:
    kernelSetup s = vector_setup();
    normKernel<<< s.grid, s.block >>>(out.data(), data(), numVectors(), dim());
    return CollOfScalar(std::move(out));
}


// ------ DOT ------
CollOfScalar CollOfVector::dot(const CollOfVector& rhs) const {
    CollOfScalar out(numVectors());
    // One thread for each vector:
    kernelSetup s = vector_setup();
    dotKernel<<< s.grid, s.block >>>( out.data(),
				      this->data(),
				      rhs.data(),
				      out.size(),
				      this->dim());
    return CollOfScalar(std::move(out));
}


const double* CollOfVector::data() const {
    return elements_.data();
}

double* CollOfVector::data() {
    return elements_.data();
}


int CollOfVector::dim() const {
    return dim_;
}

int CollOfVector::numVectors() const {
    if ( dim_ == 0 ) {
	OPM_THROW(std::runtime_error, "Calling numVectors() on a CollOfVector of dimension 0\n --> Dividing by zero!");
    }
    return elements_.size()/dim_;
}

int CollOfVector::numElements() const {
    return elements_.size();
}

kernelSetup CollOfVector::vector_setup() const {
    return kernelSetup(numVectors());
}

kernelSetup CollOfVector::element_setup() const {
    return elements_.setup();
}



//Operator []

// The output from the compiler do not use [] for Vectors (only for arrays)
// Instead we need the member function col(int) that work the same way.
// operator[] is still used in the tests, and therefore also tests col.

CollOfScalar CollOfVector::operator[](const int index) const {
    return col(index);
}

CollOfScalar CollOfVector::col(const int index) const {
    
    if ( index < 0 || index >= dim_) {
	OPM_THROW(std::runtime_error, "Illigal dimension index " << index << " for a vector of dimension " << dim_);
    }
    
    CollOfScalar out(numVectors());
    kernelSetup s = vector_setup();
    collOfVectorOperatorIndexKernel<<<s.grid,s.block>>>( out.data(),
							 this->data(),
							 out.size(),
							 index,
							 dim_);	
    return CollOfScalar(std::move(out));
}

__global__ void wrapCollOfVector::collOfVectorOperatorIndexKernel( double* out,
								   const double* vec,
								   const int size_out,
								   const int index,
								   const int dim)
{
    // Index:
    const int i = myID();
    if ( i < size_out ) {
	out[i] = vec[i*dim + index];
    }
}



// --------- NORM KERNEL ---------------------

__global__ void wrapCollOfVector::normKernel( double* out,
					      const double* vectors,
					      const int numVectors,
					      const int dim)
{
    const int index = myID();
    if ( index < numVectors ){
	double norm = 0;
	for ( int i = 0; i < dim; i++) {
	    norm += vectors[index*dim + i]*vectors[index*dim + i];
	}
	out[index] = sqrt(norm);
    }
}

// --------- DOT KERNEL ---------------------------

__global__ void wrapCollOfVector::dotKernel( double* out,
					     const double* lhs,
					     const double* rhs,
					     const int numVectors,
					     const int dim)
{
    const int index = myID();
    if ( index < numVectors ) {
	double dot = 0.0;
	for ( int i = 0; i < dim; ++i ) {
	    dot += lhs[index*dim + i] * rhs[index*dim + i];
	}
	out[index] = dot;
    }
}

// ------------- OPERATOR OVERLOADING --------------------

CollOfVector equelleCUDA::operator+(const CollOfVector& lhs, const CollOfVector& rhs) {

    CollOfVector out = lhs;
    kernelSetup s = out.element_setup();
    wrapCudaArray::plus_kernel<<<s.grid, s.block>>>(out.data(), rhs.data(), out.numElements());
    return out;
}


CollOfVector equelleCUDA::operator-(const CollOfVector& lhs, const CollOfVector& rhs) {
    CollOfVector out = lhs;
    kernelSetup s = out.element_setup();
    wrapCudaArray::minus_kernel<<<s.grid, s.block>>>(out.data(), rhs.data(), out.numElements());
    return out;
}

CollOfVector equelleCUDA::operator-(const CollOfVector& arg) {
    return (-1.0)*arg;
}

CollOfVector equelleCUDA::operator*(const Scalar lhs, const CollOfVector& rhs) {
    CollOfVector out = rhs;
    kernelSetup s = out.element_setup();
    wrapCudaArray::scalMultColl_kernel<<<s.grid, s.block>>>(out.data(), lhs, out.numElements());
    return out;
}

CollOfVector equelleCUDA::operator*(const CollOfVector& lhs, const Scalar rhs) {
    return rhs*lhs;
}

CollOfVector equelleCUDA::operator*(const CollOfVector& vec, const CollOfScalar& scal) {
    CollOfVector out = vec;
    kernelSetup s = out.vector_setup();
    collvecMultCollscal_kernel<<<s.grid, s.block>>>( out.data(),
						     scal.data(),
						     out.numVectors(),
						     out.dim());
    return out;
}

CollOfVector equelleCUDA::operator*(const CollOfScalar& scal, const CollOfVector& vec) {
    CollOfVector out = vec;
    kernelSetup s = out.vector_setup();
    collvecMultCollscal_kernel<<<s.grid, s.block>>>( out.data(),
						     scal.data(),
						     out.numVectors(),
						     out.dim());
    return out;
}

CollOfVector equelleCUDA::operator/(const CollOfVector& vec, const CollOfScalar& scal) {
    CollOfVector out = vec;
    kernelSetup s = out.vector_setup();
    collvecDivCollscal_kernel<<<s.grid, s.block>>>( out.data(),
						    scal.data(),
						    out.numVectors(),
						    out.dim());
    return out;
}

CollOfVector equelleCUDA::operator/(const CollOfVector& vec, const Scalar scal) {
    return (1.0/scal)*vec;
}



// KERNELS
__global__ void wrapCollOfVector::collvecMultCollscal_kernel( double* vector,
							      const double* scal,
							      const int numVectors,
							      const int dim)
{
    int vec = myID();
    if ( vec < numVectors ) {
	for ( int i = 0; i < dim; ++i ) {
	    vector[vec*dim + i] *= scal[vec];
	}
    }
}

__global__ void wrapCollOfVector::collvecDivCollscal_kernel( double* vector,
							     const double* scal,
							     const int numVectors,
							     const int dim)
{
    int vec = myID();
    if ( vec < numVectors ) {
	for ( int i = 0; i < dim; i++ ) {
	    vector[vec*dim + i] = vector[vec*dim + i] / scal[vec];
	}
    }

}
