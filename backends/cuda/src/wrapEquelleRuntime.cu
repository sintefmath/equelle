#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/detail/raw_pointer_cast.h>

#include "wrapEquelleRuntime.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
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

thrust::device_vector<int> equelleCUDA::trinaryIfWrapper(const CollOfBool& predicate,
							 const thrust::device_vector<int>& iftrue,
							 const thrust::device_vector<int>& iffalse) {
    thrust::device_vector<int> out(predicate.size());
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
    const int* iftrue_ptr = thrust::raw_pointer_cast( &iftrue[0] );
    const int* iffalse_ptr = thrust::raw_pointer_cast( &iffalse[0] );
    dim3 block(MAX_THREADS);
    dim3 grid((int)( (iftrue.size() + MAX_THREADS - 1)/MAX_THREADS));
    trinaryIfKernel<<<grid,block>>>( out_ptr,
				     pred_ptr,
				     iftrue_ptr,
				     iffalse_ptr,
				     iftrue.size());
    return out;
}



__global__ void equelleCUDA::trinaryIfKernel( int* out,
					      const bool* predicate,
					      const int* iftrue,
					      const int* iffalse,
					      const int size) {
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    if ( index < size ) {
	int temp;
	if ( predicate[index] ) {
	    temp = iftrue[index];
	}
	else {
	    temp = iffalse[index];
	}
	out[index] = temp;
    }
}


// Gradient implementation:
CollOfScalar equelleCUDA::gradientWrapper( const CollOfScalar& cell_scalarfield,
					   const CollOfFace& int_faces,
					   const int* face_cells) {

    // Output will be a collection on interiorFaces:
    CollOfScalar out(int_faces.size());
    // out now have info of how big kernel we need as well.
    dim3 block(out.block());
    dim3 grid(out.grid());
    gradientKernel<<<grid,block>>>( out.data(),
				    cell_scalarfield.data(),
				    int_faces.raw_pointer(),
				    face_cells,
				    out.size());
    return out;
}



__global__ void equelleCUDA::gradientKernel( double* grad,
					     const double* cell_vals,
					     const int* int_faces,
					     const int* face_cells,
					     const int size_out)
{
    // Compute index in interior_faces:
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if ( i < size_out ) {
	// Compute face index:
	int fi = int_faces[i];
	//grad[i] = second[int_face[i]] - first[int_face[i]]
	grad[i] = cell_vals[face_cells[fi*2 + 1]] - cell_vals[face_cells[fi*2]];
    }
}