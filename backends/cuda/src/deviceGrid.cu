#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>
#include <vector>

// For error exception macro:
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/grid.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/memory.h>

#include "deviceGrid.hpp"

using namespace equelleCUDA;


// -------------------------------------------------- //
// ------- Implementation of Collection ------------- //
// -------------------------------------------------- //


Collection::Collection() 
    : thrust::device_vector<int>(),
      full_(false)
{
}


Collection::Collection(const bool full)
    : thrust::device_vector<int>(),
      full_(full)
{
    if (full_ != true ) {
	OPM_THROW(std::runtime_error, "Creating non-full Collection without giving the collection\n");
    }
}


Collection::Collection(const thrust::device_vector<int>& indices) 
    : thrust::device_vector<int>(indices.begin(), indices.end()),
      full_(false)
{
}


Collection::Collection(const Collection& coll)
    : thrust::device_vector<int>(coll.begin(), coll.end()),
    full_(coll.full_)
{
}

Collection::~Collection() 
{
    // The destructor do nothing. Automaticly calling base constructor.
}

bool Collection::isFull() const
{
    return full_;
}

thrust::host_vector<int> Collection::toHost() const {
    return thrust::host_vector<int>(this->begin(), this->end());
}





// --------------------------------------------------- //
// -------- Implementation of DeviceGrid ------------- //
// --------------------------------------------------- //


// Default constructor
DeviceGrid::DeviceGrid()
    : dimensions_(0),
      number_of_cells_(0),
      number_of_faces_(0),
      cell_centroids_(0),
      cell_facepos_(0),
      cell_faces_(0),
      cell_volumes_(0),
      face_areas_(0),
      face_cells_(0),
      face_normals_(0)
{
    // intentionally left blank
}

// Constructor from a OPM UnstructuredGrid struct
DeviceGrid::DeviceGrid( const UnstructuredGrid& grid)
    : dimensions_(grid.dimensions),
      number_of_cells_(grid.number_of_cells),
      number_of_faces_(grid.number_of_faces),
      cell_centroids_(0),
      cell_facepos_(0),
      cell_faces_(0),
      cell_volumes_(0),
      face_areas_(0),
      face_cells_(0),
      face_normals_(0)
{
    // Allocate memory for cell_centroids_:
    // type: double
    // size: dimensions_ * number_of_cells_
    cudaStatus_ = cudaMalloc( (void**)&cell_centroids_ ,
			      dimensions_ * number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_centroids) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_centroids_, grid.cell_centroids,
			      dimensions_ * number_of_cells_ * sizeof(double),
			      cudaMemcpyHostToDevice );
    checkError_("cudaMemcpy(cell_centroids) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");

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
			      grid.cell_facepos[number_of_cells_] * sizeof(int));
    checkError_("cudaMalloc(cell_faces_) in DeviceGrid::DeviceGrid(UnstructuredGrid&)");
    cudaStatus_ = cudaMemcpy( cell_faces_, grid.cell_faces,
			      grid.cell_facepos[number_of_cells_] * sizeof(int),
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
}


// Destructor
DeviceGrid::~DeviceGrid() {
    if( cell_centroids_ != 0 ) {
	cudaStatus_ = cudaFree(cell_centroids_);
	checkError_("cudaFree(cell_centroids) in DeviceGrid::~DeviceGrid()");
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
}


int DeviceGrid::test() {
    return 4;
}






// Check if for CUDA error and throw OPM exception if there is one.
void DeviceGrid::checkError_(const std::string& msg) const {
    if ( cudaStatus_ != cudaSuccess ) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus_));
    }
        
}
