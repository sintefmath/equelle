
#include <cuda.h>
#include <cuda_runtime.h>

#include <opm/core/utility/ErrorMacros.hpp>
#include <iostream>
#include <vector>


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


// Copy-constructor
CollOfVector::CollOfVector(const CollOfVector& coll)
    : CollOfScalar(coll), dim_(coll.dim_)
{
    // intentionally left blank
}
  

//Operator []
CollOfScalar CollOfVector::operator[](const int index) const {
    
    if ( index < 0 || index >= dim_) {
	OPM_THROW(std::runtime_error, "Illigal dimension index " << index << " for a vector of dimension " << dim_);
    }
    
    CollOfScalar out(size()/dim_);
    
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