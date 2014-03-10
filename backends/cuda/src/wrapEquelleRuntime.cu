#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/detail/raw_pointer_cast.h>

#include "wrapEquelleRuntime.hpp"
#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"

using namespace equelleCUDA;

// Have already performed a check on sizes.
CollOfScalar equelleCUDA::trinaryIfWrapper( const CollOfBool& predicate,
					    const CollOfScalar& iftrue,
					    const CollOfScalar& iffalse) {
    CollOfScalar out(iftrue.size());
    const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
    dim3 block(iftrue.block());
    dim3 grid(iftrue.grid());
    trinaryIfKernel<<<grid,block>>>(out.data(),
				    pred_ptr,
				    iftrue.data(),
				    iffalse.data(),
				    iftrue.size());
    return out;
    //return CollOfScalar(predicate.size(), 0);
}


__global__ void equelleCUDA::trinaryIfKernel( double* out,
					      const bool* predicate,
					      const double* iftrue,
					      const double* iffalse,
					      const int size) 
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < size) {
	double temp;
	if (predicate[index]) {
	    temp = iftrue[index];
	}
	else {
	    temp = iffalse[index];
	}
	out[index] = temp;
    }
}