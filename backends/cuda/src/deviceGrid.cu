#include <cuda.h>
#include <cuda_runtime.h>

//#include "EquelleRuntimeCUDA.hpp"

#include <iostream>
#include <string>
#include <vector>

// For error exception macro:
#include <opm/common/ErrorMacros.hpp>
#include <opm/grid/GridManager.hpp>
#include <opm/grid/UnstructuredGrid.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/memory.h>
#include <thrust/fill.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/retag.h>

#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "CollOfVector.hpp"
#include "equelleTypedefs.hpp"
#include "device_functions.cuh"


using namespace equelleCUDA;
using namespace wrapDeviceGrid;


// --------------------------------------------------- //
// -------- Implementation of DeviceGrid ------------- //
// --------------------------------------------------- //


// Default constructor
DeviceGrid::DeviceGrid()
    : dimensions_(0),
      number_of_cells_(0),
      number_of_faces_(0),
      size_cell_faces_(0),
      cell_centroids_(0),
      face_centroids_(0),
      cell_facepos_(0),
      cell_faces_(0),
      cell_volumes_(0),
      face_areas_(0),
      face_cells_(0),
      face_normals_(0),
      boundary_faces_(),
      interior_faces_(),
      boundary_cells_(),
      interior_cells_(),
      boundaryFacesEmpty_(true),
      interiorFacesEmpty_(true),
      boundaryCellsEmpty_(true),
      interiorCellsEmpty_(true)
{
    // intentionally left blank
}

// Constructor from a OPM UnstructuredGrid struct
DeviceGrid::DeviceGrid( const UnstructuredGrid& grid)
    : dimensions_(grid.dimensions),
      number_of_cells_(grid.number_of_cells),
      number_of_faces_(grid.number_of_faces),
      size_cell_faces_(grid.cell_facepos[number_of_cells_]),
      cell_centroids_(0),
      face_centroids_(0),
      cell_facepos_(0),
      cell_faces_(0),
      cell_volumes_(0),
      face_areas_(0),
      face_cells_(0),
      face_normals_(0),
      boundary_faces_(),
      interior_faces_(),
      boundary_cells_(),
      interior_cells_(),
      boundaryFacesEmpty_(true),
      interiorFacesEmpty_(true),
      boundaryCellsEmpty_(true),
      interiorCellsEmpty_(true)
{
    // Allocate memory for cell_centroids_:
    // type: double
    // size: dimensions_ * number_of_cells_
    cudaStatus_ = cudaMalloc( (void**)&cell_centroids_ ,
			      dimensions_ * number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_centroids_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_centroids_, grid.cell_centroids,
			      dimensions_ * number_of_cells_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(cell_centroids_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Allocate memory for face_centroids_:
    // type: double
    // size: dimensions_ * number_of_faces_
    cudaStatus_ = cudaMalloc( (void**)&face_centroids_ ,
			      dimensions_ * number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_centroids_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( face_centroids_, grid.face_centroids,
			      dimensions_ * number_of_faces_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(face_centroids_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Allocate memory for cell_facepos_:
    // type: int
    // size: number_of_cells_ + 1
    cudaStatus_ = cudaMalloc( (void**)&cell_facepos_, 
			      (number_of_cells_ + 1) * sizeof(int));
    checkError_("cudaMalloc(cell_facepos_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_facepos_, grid.cell_facepos,
			      (number_of_cells_ + 1) * sizeof(int),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(cell_facepos_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    
    // Allocate memory for cell_faces:
    // type: int
    // size: cell_facepos_[ number_of_cells_ ]
    cudaStatus_ = cudaMalloc( (void**)&cell_faces_, 
			      size_cell_faces_ * sizeof(int));
    checkError_("cudaMalloc(cell_faces_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_faces_, grid.cell_faces,
			      size_cell_faces_ * sizeof(int),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(cell_faces_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Allocate memory for cell_volumes_:
    // type: double
    // size: number_of_cells_
    cudaStatus_ = cudaMalloc( (void**)&cell_volumes_, 
			      number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_volumes_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_volumes_, grid.cell_volumes,
			      number_of_cells_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(cell_volumes_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Allocate memory for face_areas_:
    // type: double
    // size: number_of_faces_
    cudaStatus_ = cudaMalloc( (void**)&face_areas_, 
			      number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_areas_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( face_areas_, grid.face_areas,
			      number_of_faces_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(face_areas_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
 
    // Allocate memory for face_cells_:
    // type: int
    // size: 2 * number_of_faces_
    cudaStatus_ = cudaMalloc( (void**)&face_cells_, 
			      2 * number_of_faces_ * sizeof(int));
    checkError_("cudaMalloc(face_cells_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( face_cells_, grid.face_cells,
			      2 * number_of_faces_ * sizeof(int),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(face_cells_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Allocate memory for face_normals_:
    // type: double
    // size: dimensions_ * number_of_faces_
    cudaStatus_ = cudaMalloc( (void**)&face_normals_, 
			      dimensions_ * number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_normals_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( face_normals_, grid.face_normals,
			      dimensions_ * number_of_faces_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(face_normals_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

    // Normalize the face normals. They are scaled relative to the area of the faces.
    kernelSetup s(number_of_faces_);
    normalizeAllFaceNormals<<<s.grid,s.block>>>(face_normals_, face_areas_, number_of_faces_, dimensions_);
    cudaDeviceSynchronize();

} // Constructor from OPMs UnstructuredGrid


// Copy constructor:
DeviceGrid::DeviceGrid(const DeviceGrid& grid) 
  : dimensions_(grid.dimensions_),
    number_of_cells_(grid.number_of_cells_),
    number_of_faces_(grid.number_of_faces_),
    size_cell_faces_(grid.size_cell_faces_),
    cell_centroids_(0),
    face_centroids_(0),
    cell_facepos_(0),
    cell_faces_(0),
    cell_volumes_(0),
    face_areas_(0),
    face_cells_(0),
    face_normals_(0),
    boundary_faces_(),
    interior_faces_(),
    boundary_cells_(),
    interior_cells_(),
    boundaryFacesEmpty_(true),
    interiorFacesEmpty_(true),
    boundaryCellsEmpty_(true),
    interiorCellsEmpty_(true)
{    
    // CELL_CENTROIDS_
    cudaStatus_ = cudaMalloc( (void**)&cell_centroids_,
			      dimensions_ * number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_centroids_, grid.cell_centroids_,
			      dimensions_ * number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    // FACE_CENTROIDS_
    cudaStatus_ = cudaMalloc( (void**)&face_centroids_,
			      dimensions_ * number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_centroids_, grid.face_centroids_,
			      dimensions_ * number_of_faces_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");

    // CELL_FACEPOS_
    cudaStatus_ = cudaMalloc( (void**)&cell_facepos_, 
			      (number_of_cells_ + 1) * sizeof(int));
    checkError_("cudaMalloc(cell_facepos_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_facepos_, grid.cell_facepos_,
			      (number_of_cells_ + 1) * sizeof(int),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_facepos_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");

    // CELL_FACES_
    cudaStatus_ = cudaMalloc( (void**)&cell_faces_,
			      size_cell_faces_ * sizeof(int));
    checkError_("cudaMalloc(cell_faces_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_faces_, grid.cell_faces_,
			      size_cell_faces_ * sizeof(int),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_faces_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");

    // CELL_VOLUMES_
    cudaStatus_ = cudaMalloc( (void**)&cell_volumes_,
			      number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_volumes_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_volumes_, grid.cell_volumes_,
			      number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(cell_volumes_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");

    // FACE_AREAS_
    cudaStatus_ = cudaMalloc( (void**)&face_areas_,
			      number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_areas_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_areas_, grid.face_areas_,
			      number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_areas_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    
    // FACE_CELLS_
    cudaStatus_ = cudaMalloc( (void**)&face_cells_,
			      2 * number_of_faces_ * sizeof(int));
    checkError_("cudaMalloc(face_cells_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_cells_, grid.face_cells_,
			      2 * number_of_faces_ * sizeof(int),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_cells_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
			    
    // FACE_NORMALS
    cudaStatus_ = cudaMalloc( (void**)&face_normals_,
			      dimensions_ * number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_normals_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_normals_, grid.face_normals_,
			      dimensions_ * number_of_faces_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_normals_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");

    // Normalize the face normals. They are scaled relative to the area of the faces.
    kernelSetup s(number_of_faces_);
    normalizeAllFaceNormals<<<s.grid,s.block>>>(face_normals_, face_areas_, number_of_faces_, dimensions_);
    cudaDeviceSynchronize();
} // copy constructor


// Destructor
DeviceGrid::~DeviceGrid() {

    if( cell_centroids_ != 0 ) {
	cudaStatus_ = cudaFree(cell_centroids_);
	checkError_("cudaFree(cell_centroids_) in DeviceGrid::~DeviceGrid()");
    }
    if ( face_centroids_ != 0 ) {
	cudaStatus_ = cudaFree(face_centroids_);
	checkError_("cudaFree(face_centroids) in DeviceGrid::~DeviceGrid()");
    }
    if ( cell_facepos_ != 0 ) {
	cudaStatus_ = cudaFree(cell_facepos_);
	checkError_("cudaFree(cell_facepos_) in DeviceGrid::~DeviceGrid()");
    }
    if ( cell_faces_ != 0 ) {
	cudaStatus_ = cudaFree(cell_faces_);
	checkError_("cudaFree(cell_faces_) in DeviceGrid::~DeviceGrid()");
    }
    if ( cell_volumes_ != 0 ) {
	cudaStatus_ = cudaFree(cell_volumes_);
	checkError_("cudaFree(cell_volumes_) in DeviceGrid::~DeviceGrid()");
    }
    if ( face_areas_ != 0 ) {
	cudaStatus_ = cudaFree(face_areas_);
	checkError_("cudaFree(face_areas_) in DeviceGrid::~DeviceGrid()");
    }
    if ( face_cells_ != 0 ) {
	cudaStatus_ = cudaFree(face_cells_);
	checkError_("cudaFree(face_cells_) in DeviceGrid::~DeviceGrid()");
    }
    if ( face_normals_ != 0 ) {
	cudaStatus_ = cudaFree(face_normals_);
	checkError_("cudaFree(face_normals_) in DeviceGrid::~DeviceGrid()");
    }

} // Destructor



// ------------ GRID OPERATIONS! ------------
CollOfCell DeviceGrid::allCells() const {
    return CollOfCell(number_of_cells_);
}

CollOfFace DeviceGrid::allFaces() const {
    return CollOfFace(number_of_faces_);
}


const CollOfFace& DeviceGrid::boundaryFaces() const {
    if ( boundaryFacesEmpty_ ) {
	createBoundaryFaces_();
    }
    return boundary_faces_;
}

const CollOfFace& DeviceGrid::interiorFaces() const {
    if ( interiorFacesEmpty_ ) {
	createInteriorFaces_();
    }
    return interior_faces_;
}

const CollOfCell& DeviceGrid::boundaryCells() const {
    if ( boundaryCellsEmpty_ ) {
	createBoundaryCells_();
    }
    return boundary_cells_;
}

const CollOfCell& DeviceGrid::interiorCells() const {
    if ( interiorCellsEmpty_ ) {
	createInteriorCells_();
    }
    return interior_cells_;
}


void DeviceGrid::createBoundaryFaces_() const {
    // we use the face_cells_ array to check if both face_cells are cells
    // If face f is a boundary face, then 
    // face_cells_[2 * f] or face_cells_[2 * f + 1] contains -1.
    
    // Launch a kernel where we use number_of_faces number of threads.
    // Use a 1D kernel for simplicity.
    // Assume that we do not need more blocks than available.
    kernelSetup s(number_of_faces_);

    // Create a vector of size number_of_faces_:
    thrust::device_vector<int> b_faces(number_of_faces_);
    // Fill it with the value number_of_faces_
    //     this is an illigal faca index
    thrust::fill(b_faces.begin(), b_faces.end(), number_of_faces_);
    int* b_faces_ptr = thrust::raw_pointer_cast( &b_faces[0] );
    boundaryFacesKernel<<<s.grid, s.block>>>( b_faces_ptr,
					      face_cells_,
					      number_of_faces_);
    
    // Remove unchanged values
    // See  - thrust::remove_if documentation 
    //      - the saxpy example in the algorithm chapter of the thrust pdf
    thrust::device_vector<int>::iterator new_end = thrust::remove_if(thrust::device, 
								     b_faces.begin(),
								     b_faces.end(),
								     unchanged(number_of_faces_));
    
    // new_end points now to where the legal values end,
    // but the vector still has size equal to number_of_faces_
    boundary_faces_ = CollOfFace(b_faces.begin(), new_end);
    boundaryFacesEmpty_ = false;
}


void DeviceGrid::createInteriorFaces_() const {
    // we use the face_cells_ array to check if both face_cells are cells
    // If face f is an interior face, then neither of
    // face_cells_[2 * f] nor face_cells_[2 * f + 1] contains -1.
    
    // Launch a kernel where we use number_of_faces number of threads.
    // Use a 1D kernel for simplicity.
    // Assume that we do not need more blocks than available.
     kernelSetup s(number_of_faces_);

    // Create a vector of size number_of_faces_:
    thrust::device_vector<int> i_faces(number_of_faces_);
    // Fill it with the value number_of_faces_
    //     this is an illigal faca index
    thrust::fill(i_faces.begin(), i_faces.end(), number_of_faces_);
    int* i_faces_ptr = thrust::raw_pointer_cast( &i_faces[0] );
    interiorFacesKernel<<<s.grid, s.block>>>( i_faces_ptr,
					      face_cells_,
					      number_of_faces_);
    // Remove unchanged values
    // See  - thrust::remove_if documentation 
    //      - the saxpy example in the algorithm chapter of the thrust pdf
    thrust::device_vector<int>::iterator new_end = thrust::remove_if(thrust::device, 
								     i_faces.begin(),
								     i_faces.end(),
								     unchanged(number_of_faces_));
    
    // new_end points now to where the legal values end,
    // but the vector still has size equal to number_of_faces_    
    interior_faces_ = CollOfFace(i_faces.begin(), new_end);
    interiorFacesEmpty_ = false;
}


// BOUNDARY CELLS
void DeviceGrid::createBoundaryCells_() const {
    // Returns a Collection of indices of boundary cells.
    // Algorithm:
    // for each cell c
    //     for (face f_index in [cell_facepos[c] : cell_facepos[c+1] )
    //          f = cell_faces[f_index]
    //          if ( face_cells[2*f] == -1 or face_cells[2*f + 1] == -1 )
    //              c is a boundary cell.

    // Kernel of number_of_cells_ threads
    // Operate on vector filled with number_of_cells_
    // Set cell index if boundary cell
    // Remove all elements equal to number_of_cells_.

    kernelSetup s(number_of_cells_);
    thrust::device_vector<int> b_cells(number_of_cells_);
    thrust::fill(b_cells.begin(), b_cells.end(), number_of_cells_);
    int* b_cells_ptr = thrust::raw_pointer_cast( &b_cells[0] );
    boundaryCellsKernel<<<s.grid, s.block>>>( b_cells_ptr,
					      number_of_cells_,
					      cell_facepos_,
					      cell_faces_,
					      face_cells_);

    // Remove values which still are number_of_cells_
    thrust::device_vector<int>::iterator new_end = thrust::remove_if(thrust::device,
								     b_cells.begin(),
								     b_cells.end(),
								     unchanged(number_of_cells_));
    boundary_cells_ = CollOfCell(b_cells.begin(), new_end);
    boundaryCellsEmpty_ = false;
}


// INTERIOR CELLS
void DeviceGrid::createInteriorCells_() const {
    // Same as boundaryCells, but the kernel is the other way around
    kernelSetup s(number_of_cells_);
    thrust::device_vector<int> i_cells(number_of_cells_);
    thrust::fill(i_cells.begin(), i_cells.end(), number_of_cells_);
    int* i_cells_ptr = thrust::raw_pointer_cast( &i_cells[0] );
    interiorCellsKernel<<<s.grid, s.block>>>( i_cells_ptr,
					      number_of_cells_,
					      cell_facepos_,
					      cell_faces_,
					      face_cells_);

    // Remove values which still are number_of_cells_
    thrust::device_vector<int>::iterator new_end = thrust::remove_if(thrust::device,
								     i_cells.begin(),
								     i_cells.end(),
								     unchanged(number_of_cells_));
    interior_cells_ = CollOfCell(i_cells.begin(), new_end);
    interiorCellsEmpty_ = false;
}


// FIRST AND SECOND
CollOfCell DeviceGrid::firstCell(CollOfFace coll) const {
    // The out collection will be of same size as the in collection

    // FirstCells are found from the face_cells_ array
    // for face f
    //     first(f) = face_cells_[2*f]
    
    // setup how many threads/blocks we need:
    kernelSetup s(coll.size());

    // create a vector of size number_of_faces_:
    thrust::device_vector<int> first(coll.size());
    int* first_ptr = thrust::raw_pointer_cast( &first[0] );
    if (coll.isFull()) {
	firstCellKernel<<<s.grid, s.block>>>( first_ptr, coll.size(), face_cells_);
    } else {
	int* index_ptr = coll.raw_pointer();
 	firstCellSubsetKernel<<<s.grid, s.block>>>( first_ptr, coll.size(),
						    index_ptr, face_cells_);
    }					
    return CollOfCell(first);
}

CollOfCell DeviceGrid::secondCell(CollOfFace coll) const {
    // SecondCells are found from the face_cells_ array
    // for face f
    //     second(f) = face_cells_[2*f + 1]

    // setup how many threads/blocks we need:
    kernelSetup s(coll.size());
    
    // create a vector of size number_of_faces_:
    thrust::device_vector<int> second(coll.size());
    int* second_ptr = thrust::raw_pointer_cast( &second[0] );
    if ( coll.isFull() ) {
	secondCellKernel<<<s.grid, s.block>>>( second_ptr, coll.size(), face_cells_);
    } else {
	secondCellSubsetKernel<<<s.grid, s.block>>>( second_ptr, coll.size(),
						     coll.raw_pointer(), face_cells_);
    }
    return CollOfCell(second);
}


// ----- NORM ----

CollOfScalar DeviceGrid::norm_of_cells(const thrust::device_vector<int>& cells,
				       const bool full) const {
    if (full) {
	CollOfScalar out(number_of_cells_);
	cudaStatus_ = cudaMemcpy( out.data(), cell_volumes_, 
				  sizeof(double)*number_of_cells_,
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy in DeviceGrid::norm_of_cells");
	return CollOfScalar(std::move(out));
    }
    else {
	CollOfScalar out(cells.size());
	kernelSetup s = out.setup();
	const int* cells_ptr = thrust::raw_pointer_cast( &cells[0] );
	normKernel<<<s.grid, s.block>>>( out.data(), cells_ptr, cells.size(),
					 cell_volumes_);
	return CollOfScalar(std::move(out));
    }
}

CollOfScalar DeviceGrid::norm_of_faces(const thrust::device_vector<int>& faces,
				       const bool full) const {
    if (full) {
	CollOfScalar out(number_of_faces_);
	cudaStatus_ = cudaMemcpy(out.data(), face_areas_, 
				 sizeof(double)*number_of_faces_,
				 cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy in DeviceGrid::norm_of_cells");
	return CollOfScalar(std::move(out));
    }
    else {
	CollOfScalar out(faces.size());
	kernelSetup s = out.setup();
	const int* faces_ptr = thrust::raw_pointer_cast( &faces[0] );
	normKernel<<<s.grid, s.block>>>( out.data(), faces_ptr, faces.size(),
					 face_areas_);
	return CollOfScalar(std::move(out));
    }
}


// CENTROID

CollOfVector DeviceGrid::centroid(const thrust::device_vector<int>& indices,
				  const bool full,
				  const int codim) const {
    if (full) {
	if (codim == 0) { // All cells
	    CollOfVector out(number_of_cells_, dimensions_);
	    cudaStatus_ = cudaMemcpy(out.data(), cell_centroids_,
				     sizeof(double)*dimensions_*number_of_cells_,
				     cudaMemcpyDeviceToDevice);
	    checkError_("cudaMemcpy in DeviceGrid::centroid(..) -> full -> codim=0");
	    return CollOfVector(std::move(out));
	}
	else { // All faces
	    CollOfVector out(number_of_faces_, dimensions_);
	    cudaStatus_ = cudaMemcpy(out.data(), face_centroids_,
				     sizeof(double)*dimensions_*number_of_faces_,
				     cudaMemcpyDeviceToDevice);
	    checkError_("cudaMemcpy in DeviceGrid::centroids(..) -> full -> codim=1");
	    return CollOfVector(std::move(out));
	}
    }
    else {
	CollOfVector out(indices.size(), dimensions_);
	// Set up a kernel to find the subset
	// Easy implementation: 
	// CollOfVector::block() and grid() assumes one thread per double value
	// Our kernel use one thread per vector, so we overshoot a bit.
	kernelSetup s = out.element_setup();
	const int* indices_ptr = thrust::raw_pointer_cast( &indices[0] );
	
	// Get a pointer to the correct set of centroids:
	const double* all_centroids = cell_centroids_;
	if ( codim == 1) {
	    all_centroids = face_centroids_;
	}
	centroidKernel<<<s.grid, s.block>>>( out.data(),
					     indices_ptr,
					     all_centroids,
					     out.numVectors(),
					     dimensions_);
	return CollOfVector(std::move(out));
    }
}


// NORMAL:
CollOfVector DeviceGrid::normal( const CollOfFace& faces) const {
    CollOfVector out(faces.size(), dimensions_);
    // cudaMemcpy to get the normals if the set is full.
    if ( faces.isFull() ) {
	
	cudaStatus_ = cudaMemcpy(out.data(), face_normals_,
				 sizeof(double)*out.numVectors()*dimensions_,
				 cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(face_normals) in DeviceGrid::normal(const CollOfFaces&)");
    }
    else {
	// Need a Kernel to fetch only the correct ones.
	// Easy implementation:
	// CollOfVector::block() and grid() assumes one thread per double value
	// Our kernel use one thread per vector, so we overshoot a bit.
	kernelSetup s = out.element_setup();
	faceNormalsKernel<<<s.grid, s.block>>>(out.data(),
					       faces.raw_pointer(),
					       face_normals_,
					       out.numVectors(),
					       dimensions_);
    }
    return CollOfVector(std::move(out));
}



// ----------- GET FUNCTIONS! ------------------

int DeviceGrid::dimensions() const {
    return dimensions_;
}

int DeviceGrid::number_of_cells() const {
    return number_of_cells_;
}

int DeviceGrid::number_of_faces() const {
    return number_of_faces_;
}

int* DeviceGrid::cell_facepos() const {
    return cell_facepos_;
}

int* DeviceGrid::cell_faces() const {
    return cell_faces_;
}

int* DeviceGrid::face_cells() const {
    return face_cells_;
}

// ---------- ERROR CHECKING! -----------------------

// Check if for CUDA error and throw OPM exception if there is one.
void DeviceGrid::checkError_(const std::string& msg) const {
    if ( cudaStatus_ != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus_));
    }        
}




// ----------- GRID KERNELS -------------------------

__global__ void wrapDeviceGrid::boundaryFacesKernel( int* b_faces,
						     const int* face_cells,
						     const int number_of_faces) 
{
    const int face = myID();
    if (face < number_of_faces) {
	if ( (face_cells[2*face] == -1) || (face_cells[2*face + 1] == -1) ) {
	    b_faces[face] = face;
	}
    }
}


__global__ void wrapDeviceGrid::interiorFacesKernel( int* i_faces,
						     const int* face_cells,
						     const int number_of_faces)
{
    const int face = myID();
    if ( face < number_of_faces) {
	if ( (face_cells[2*face] != -1) && (face_cells[2*face + 1] != -1) ) {
	    i_faces[face] = face;
	}
    }
}


__global__ void wrapDeviceGrid::boundaryCellsKernel(int* b_cells,
						    const int number_of_cells,
						    const int* cell_facepos,
						    const int* cell_faces,
						    const int* face_cells)
{
    const int cell = myID();
    if ( cell < number_of_cells) {
	bool boundary = false;
	int face;
	for ( int f_i = cell_facepos[cell]; f_i < cell_facepos[cell + 1]; f_i++) {
	    face = cell_faces[f_i];
	    if ( (face_cells[ 2*face ] == -1) || (face_cells[ 2*face +1] == -1) ) {
		boundary = true;
	    }
	}
	if (boundary) {
	    b_cells[cell] = cell;
	}
    }
}


__global__ void wrapDeviceGrid::interiorCellsKernel( int* i_cells,
						     const int number_of_cells,
						     const int* cell_facepos,
						     const int* cell_faces,
						     const int* face_cells)
{
    const int cell = myID();
    if ( cell < number_of_cells) {
	bool interior = true;
	int face;
	for ( int f_i = cell_facepos[cell]; f_i < cell_facepos[cell + 1]; f_i++) {
	    face = cell_faces[f_i];
	    if ( (face_cells[ 2*face ] == -1) || (face_cells[ 2*face +1] == -1) ) {
		interior = false;
	    }
	}
	if ( interior ) {
	    i_cells[cell] = cell;
	}
    }

}


__global__ void wrapDeviceGrid::firstCellKernel( int* first,
						 const int number_of_faces,
						 const int* face_cells)
{
    // For face f:
    //     first(f) = face_cells[2*f]
    const int face = myID();
    if ( face < number_of_faces ) {
	first[face] = face_cells[2*face];
    }
}

__global__ void wrapDeviceGrid::firstCellSubsetKernel( int* first,
						       const int number_of_faces,
						       const int* face_index,
						       const int* face_cells)
{
    // For thread i:
    //      first(i) = face_cells[2*face_index[i]]
    const int index = myID();
    if ( index < number_of_faces ) {
	first[index] = face_cells[2*face_index[index]];
    }
}

__global__ void wrapDeviceGrid::secondCellKernel( int* second,
						  const int number_of_faces,
						  const int* face_cells)
{
    // For face f:
    //     second(f) = face_cells[2*f + 1]
    const int face = myID();
    if ( face < number_of_faces ) {
	second[face] = face_cells[2*face + 1];
    }
 }

__global__ void wrapDeviceGrid::secondCellSubsetKernel( int* second,
							const int number_of_faces,
							const int* face_index,
							const int* face_cells)
{
    // for thread i
    //     second[i] = face_cells[2*face_index[i] + 1]
    const int index = myID();
    if ( index < number_of_faces ) {
	second[index] = face_cells[2*face_index[index] + 1];
    }
}


// NORM KERNEL


__global__ void wrapDeviceGrid::normKernel( double* out,
					    const int* indices,
					    const int out_size,
					    const double* norm_values) 
{
    const int index = myID();
    if ( index < out_size ) {
	out[index] = norm_values[indices[index]];
    }
}

// CENTROID KERNEL

__global__ void wrapDeviceGrid::centroidKernel( double* out,
						const int* subset_indices,
						const double* all_centroids,
						const int num_vectors,
						const int dimensions)
{
    // EASY IMPLEMENTATION:
    // One thread for each vector
    const int vec_id = myID();
    if ( vec_id < num_vectors ) {
	const int cell_index = subset_indices[vec_id];
	// Iterating over the element in the vector we create
	for (int i = 0; i < dimensions; i++) {
	    out[vec_id*dimensions + i] = all_centroids[cell_index * dimensions + i];
	}
    }
}



// FACE NORMALS
__global__ void wrapDeviceGrid::faceNormalsKernel( double* out,
						   const int* faces,
						   const double* all_face_normals,
						   const int num_vectors,
						   const int dimensions)
{
    // EASY IMPLEMENTATION
    // One thread for each vector
    const int vec_id = myID();
    if ( vec_id < num_vectors ) {
	const int face_id = faces[vec_id];
	for( int i = 0; i < dimensions; i++) {
	    out[vec_id*dimensions + i] = all_face_normals[face_id*dimensions + i];
	}
    }
}



// NORMALIZE FACE NORMALS
__global__ void wrapDeviceGrid::normalizeAllFaceNormals( double* normals, 
                                                         const double* face_areas_,
                                                         const int num_vectors,
                                                         const int dimensions)
{
    // One thread for each vector
    const int face_id = myID();
    if ( face_id < num_vectors ) {
        if(dimensions == 1){
            normals[face_id] /= face_areas_[face_id];
        } else
        if(dimensions == 2){
            normals[face_id] /= face_areas_[face_id];
            normals[face_id*dimensions+1] /= face_areas_[face_id];
        } else
        if(dimensions == 3) {
            normals[face_id*dimensions]   /= face_areas_[face_id];
            normals[face_id*dimensions+1] /= face_areas_[face_id];
            normals[face_id*dimensions+2] /= face_areas_[face_id];
        }
    }
}