#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>


#include <math.h>
#include <iostream>
#include <limits>

#include <opm/common/ErrorMacros.hpp>

#include "wrapEquelleRuntime.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "DeviceGrid.hpp"
#include "device_functions.cuh"
#include "DeviceHelperOps.hpp"

using namespace equelleCUDA;
using namespace wrapEquelleRuntimeCUDA;



// Declaring the cuSparse handle!
cusparseHandle_t equelleCUDA::CUSPARSE;

void wrapEquelleRuntimeCUDA::init_cusparse() {
    cusparseStatus_t status;
    status = cusparseCreate(&CUSPARSE);
    if (status != CUSPARSE_STATUS_SUCCESS ) {
	OPM_THROW(std::runtime_error, "Cannot create cusparse handle.");
    }
}

void wrapEquelleRuntimeCUDA::destroy_cusparse() {
    cusparseStatus_t status;
    status = cusparseDestroy(CUSPARSE);
    if (status != CUSPARSE_STATUS_SUCCESS ) {
	OPM_THROW(std::runtime_error, "Cannot destroy cusparse handle.");
    }
}


// --------------  TRINARY IF -----------------------

// Have already performed a check on sizes.
CollOfScalar wrapEquelleRuntimeCUDA::trinaryIfWrapper( const CollOfBool& predicate,
						       const CollOfScalar& iftrue,
						       const CollOfScalar& iffalse) {
    if ( iftrue.useAutoDiff() || iffalse.useAutoDiff()) {
	CudaArray val(iftrue.size());
	const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
	kernelSetup s = val.setup();
	trinaryIfKernel<<<s.grid, s.block>>>(val.data(),
					     pred_ptr,
					     iftrue.data(),
					     iffalse.data(),
					     iftrue.size());
	// Using matrix-multiplication for derivatives
	CudaMatrix diagBool(predicate);
	CudaMatrix der = diagBool*iftrue.derivative() + (CudaMatrix(predicate.size()) - diagBool)*iffalse.derivative();
	
	return CollOfScalar(std::move(val), std::move(der));
    }
    else { // No AutoDiff
	CollOfScalar out(iftrue.size());
	const bool* pred_ptr = thrust::raw_pointer_cast( &predicate[0] );
	kernelSetup s = out.setup();
	trinaryIfKernel<<<s.grid, s.block>>>(out.data(),
					     pred_ptr,
					     iftrue.data(),
					     iffalse.data(),
					     iftrue.size());
	return CollOfScalar(std::move(out));
    }
}


// For indicis

__global__ void wrapEquelleRuntimeCUDA::trinaryIfKernel( double* out,
							 const bool* predicate,
							 const double* iftrue,
							 const double* iffalse,
							 const int size) 
{
    const int index = myID();
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
    const int index = myID();
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


// ----------------- GRADIENT ----------------------

// Gradient implementation:
CollOfScalar wrapEquelleRuntimeCUDA::gradientWrapper( const CollOfScalar& cell_scalarfield,
						      const CollOfFace& int_faces,
						      const int* face_cells,
						      DeviceHelperOps& ops) {
    // This function is at the moment kept in order to be able to compare efficiency
    // against the new implementation, where we use the matrix from devOps_.

    if ( cell_scalarfield.useAutoDiff() ) {
	// Output will be a collection on interiorFaces:
	CudaArray val(int_faces.size());
	// out now have info of how big kernel we need as well.
	kernelSetup s = val.setup();
	gradientKernel<<<s.grid, s.block>>>( val.data(),
					     cell_scalarfield.data(),
					     int_faces.raw_pointer(),
					     face_cells,
					     val.size());
	CudaMatrix der = ops.grad() * cell_scalarfield.derivative();
	return CollOfScalar(std::move(val), std::move(der));
    }
    else {
	CollOfScalar out(int_faces.size());
	kernelSetup s = out.setup();
	gradientKernel<<<s.grid, s.block>>>( out.data(),
					     cell_scalarfield.data(),
					     int_faces.raw_pointer(),
					     face_cells,
					     out.size());
	
	return CollOfScalar(std::move(out));
    }
}



__global__ void wrapEquelleRuntimeCUDA::gradientKernel( double* grad,
							const double* cell_vals,
							const int* int_faces,
							const int* face_cells,
							const int size_out)
{
    // Compute index in interior_faces:
    const int i = myID();
    if ( i < size_out ) {
	// Compute face index:
	const int fi = int_faces[i];
	//grad[i] = second[int_face[i]] - first[int_face[i]]
	grad[i] = cell_vals[face_cells[fi*2 + 1]] - cell_vals[face_cells[fi*2]];
    }
}



// ------------- DIVERGENCE --------------- //

CollOfScalar wrapEquelleRuntimeCUDA::divergenceWrapper( const CollOfScalar& fluxes,
							const DeviceGrid& dev_grid,
							DeviceHelperOps& ops) {
    if ( fluxes.useAutoDiff() ) {
	CudaArray val(dev_grid.number_of_cells());
	kernelSetup s = val.setup();
	divergenceKernel<<<s.grid, s.block>>>( val.data(),
					       fluxes.data(),
					       dev_grid.cell_facepos(),
					       dev_grid.cell_faces(),
					       dev_grid.face_cells(),
					       dev_grid.number_of_cells(),
					       dev_grid.number_of_faces() );
	CudaMatrix der = ops.fulldiv() * fluxes.derivative();
	return CollOfScalar(std::move(val), std::move(der));	
    }

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

    return CollOfScalar(std::move(out));
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
    const int cell = myID();
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
    
    CudaArray val = x.value();
    kernelSetup s = val.setup();
    sqrtKernel<<<s.grid, s.block>>> (val.data(), val.size());
    if ( x.useAutoDiff() ) {
	// sqrt(x)' = 1/(2*sqrt(x)) * x'
	CudaMatrix diag(1/(2*val));
	CudaMatrix der = diag * x.derivative();
	return CollOfScalar(std::move(val), std::move(der));
    }
    return CollOfScalar(std::move(val));
}


__global__ void wrapEquelleRuntimeCUDA::sqrtKernel(double* out, const int size) {
    const int index = myID();
    if ( index < size ) {
	out[index] = sqrt(out[index]);
    }
}



