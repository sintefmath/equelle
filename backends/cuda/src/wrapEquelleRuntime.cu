#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <math.h>

#include "wrapEquelleRuntime.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "DeviceGrid.hpp"

using namespace equelleCUDA;
using namespace wrapEquelleRuntimeCUDA;

// Declaring the cuSparse handle!
cusparseHandle_t equelleCUDA::CUSPARSE;


// Have already performed a check on sizes.
CollOfScalar wrapEquelleRuntimeCUDA::trinaryIfWrapper( const CollOfBool& predicate,
						       const CollOfScalar& iftrue,
						       const CollOfScalar& iffalse) {
    CollOfScalar out(iftrue.size());
    const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
    kernelSetup s = out.setup();
    trinaryIfKernel<<<s.grid, s.block>>>(out.data(),
					 pred_ptr,
					 iftrue.data(),
					 iffalse.data(),
					 iftrue.size());
    return out;
}


__global__ void wrapEquelleRuntimeCUDA::trinaryIfKernel( double* out,
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


thrust::device_vector<int> 
wrapEquelleRuntimeCUDA::trinaryIfWrapper(const CollOfBool& predicate,
					 const thrust::device_vector<int>& iftrue,
					 const thrust::device_vector<int>& iffalse) {
    thrust::device_vector<int> out(predicate.size());
    int* out_ptr = thrust::raw_pointer_cast( &out[0] );
    const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
    const int* iftrue_ptr = thrust::raw_pointer_cast( &iftrue[0] );
    const int* iffalse_ptr = thrust::raw_pointer_cast( &iffalse[0] );
    kernelSetup s(iftrue.size());
    trinaryIfKernel<<<s.grid, s.block>>>( out_ptr,
					  pred_ptr,
					  iftrue_ptr,
					  iffalse_ptr,
					  iftrue.size());
    return out;
}



__global__ void wrapEquelleRuntimeCUDA::trinaryIfKernel( int* out,
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
CollOfScalar wrapEquelleRuntimeCUDA::gradientWrapper( const CollOfScalar& cell_scalarfield,
						      const CollOfFace& int_faces,
						      const int* face_cells) {
    
    // Output will be a collection on interiorFaces:
    CollOfScalar out(int_faces.size());
    // out now have info of how big kernel we need as well.
    kernelSetup s = out.setup();
    gradientKernel<<<s.grid, s.block>>>( out.data(),
					 cell_scalarfield.data(),
					 int_faces.raw_pointer(),
					 face_cells,
					 out.size());
    return out;
}



__global__ void wrapEquelleRuntimeCUDA::gradientKernel( double* grad,
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

CollOfScalar wrapEquelleRuntimeCUDA::divergenceWrapper( const CollOfScalar& fluxes,
							const DeviceGrid& dev_grid) {

    // output is of size number_of_cells:
    CollOfScalar out(dev_grid.number_of_cells());
    // out have now block and grid size as well.
    kernelSetup s = out.setup();
    divergenceKernel<<<s.grid, s.block>>>( out.data(),
					   fluxes.data(),
					   dev_grid.cell_facepos(),
					   dev_grid.cell_faces(),
					   dev_grid.face_cells(),
					   dev_grid.number_of_cells(),
					   dev_grid.number_of_faces() );

    return out;
}


__global__ void wrapEquelleRuntimeCUDA::divergenceKernel( double* div,
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


// --------- SQRT ----------------
CollOfScalar wrapEquelleRuntimeCUDA::sqrtWrapper( const CollOfScalar& x) {
    
    CollOfScalar out = x;
    kernelSetup s = out.setup();
    sqrtKernel<<<s.grid, s.block>>> (out.data(), out.size());
    return out;
}


__global__ void wrapEquelleRuntimeCUDA::sqrtKernel(double* out, const int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if ( index < size ) {
	out[index] = sqrt(out[index]);
    }
}