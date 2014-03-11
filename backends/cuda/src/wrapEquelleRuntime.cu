#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/detail/raw_pointer_cast.h>

#include "wrapEquelleRuntime.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "DeviceGrid.hpp"

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



// ------------- DIVERGENCE --------------- //

CollOfScalar equelleCUDA::divergenceWrapper( const CollOfScalar& fluxes,
					     const DeviceGrid& dev_grid) {

    // output is of size number_of_cells:
    CollOfScalar out(dev_grid.number_of_cells());
    // out have now block and grid size as well.
    dim3 block(out.block());
    dim3 grid(out.grid());

    divergenceKernel<<<grid,block>>>( out.data(),
				      fluxes.data(),
				      dev_grid.cell_facepos(),
				      dev_grid.cell_faces(),
				      dev_grid.face_cells(),
				      dev_grid.number_of_cells(),
				      dev_grid.number_of_faces() );

    return out;
}


__global__ void equelleCUDA::divergenceKernel( double* div,
					       const double* flux,
					       const int* cell_facepos,
					       const int* cell_faces,
					       const int* face_cells,
					       const int number_of_cells,
					       const int number_of_faces) 
{
    // My index: cell
    int cell = threadIdx.x + blockIdx.x*blockDim.x;
    if ( cell < number_of_cells ) {
	double div_temp = 0; // total divergence for this cell.
	int factor, face;
	// Iterate over this cells faces:
	for ( int i = cell_facepos[cell]; i < cell_facepos[cell+1]; ++i ) {
	    factor = -1; // Assume normal inwards
	    face = cell_faces[i];
	    if ( face_cells[face*2] == cell ) { // if normal outwards
		factor = 1;
	    }
	    // Add contribution from this cell
	    div_temp += flux[face]*factor; 
	}
	div[cell] = div_temp;
    }
}