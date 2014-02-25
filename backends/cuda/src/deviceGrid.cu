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
      size_cell_faces_(0),
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
      size_cell_faces_(grid.cell_facepos[number_of_cells_]),
      cell_centroids_(0),
      cell_facepos_(0),
      cell_faces_(0),
      cell_volumes_(0),
      face_areas_(0),
      face_cells_(0),
      face_normals_(0)
{
    std::cout << "Creating new DeviceGrid:\n";
    std::cout << "\t\t Dimensions = " << dimensions_ << std::endl;
    std::cout << "\t\t Number of cells = " << number_of_cells_ << std::endl;
    std::cout << "\t\t Number of faces = " << number_of_faces_ << std::endl;
    std::cout << "\t\t size of cell_faces_ = " << size_cell_faces_ << std::endl;

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
    std::cout << "\tnew cell_centroids_ " << cell_centroids_ << "\n";

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
    std::cout << "\tnew cell_facepos_ " << cell_facepos_ << "\n";
    
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
    std::cout << "\tnew cell_faces_ " << cell_faces_ << " with size " << size_cell_faces_ << "\n";

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
    std::cout << "\tnew cell_volumes_ " << cell_volumes_ << "\n";

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

    std::cout << "Created DeviceGrid from UnstructuredGrid!\n";
}


// Copy constructor:
DeviceGrid::DeviceGrid(const DeviceGrid& grid) 
  : dimensions_(grid.dimensions_),
    number_of_cells_(grid.number_of_cells_),
    number_of_faces_(grid.number_of_faces_),
    size_cell_faces_(grid.size_cell_faces_),
    cell_centroids_(0),
    cell_facepos_(0),
    cell_faces_(0),
    cell_volumes_(0),
    face_areas_(0),
    face_cells_(0),
    face_normals_(0)
{    
    // CELL_CENTROIDS_
    cudaStatus_ = cudaMalloc( (void**)&cell_centroids_,
			      dimensions_ * number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_centroids_, grid.cell_centroids_,
			      dimensions_ * number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_centroids_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy cell_centroids_:\t" << cell_centroids_ << "\n";

    // CELL_FACEPOS_
    cudaStatus_ = cudaMalloc( (void**)&cell_facepos_, 
			      (number_of_cells_ + 1) * sizeof(int));
    checkError_("cudaMalloc(cell_facepos_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_facepos_, grid.cell_facepos_,
			      (number_of_cells_ + 1) * sizeof(int),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_facepos_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy cell_facepos_:\t" <<cell_facepos_ << "\n";

    // CELL_FACES_
    cudaStatus_ = cudaMalloc( (void**)&cell_faces_,
			      size_cell_faces_ * sizeof(int));
    checkError_("cudaMalloc(cell_faces_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_faces_, grid.cell_faces_,
			      size_cell_faces_ * sizeof(int),
			      cudaMemcpyDeviceToDevice );
    checkError_("cudaMemcpy(cell_faces_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy cell_faces_:\t" << cell_faces_ << " with size ";
    std::cout << size_cell_faces_ << "\n";

    // CELL_VOLUMES_
    cudaStatus_ = cudaMalloc( (void**)&cell_volumes_,
			      number_of_cells_ * sizeof(double));
    checkError_("cudaMalloc(cell_volumes_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( cell_volumes_, grid.cell_volumes_,
			      number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(cell_volumes_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy cell_volumes_:\t" << cell_volumes_ << "\n";

    // FACE_AREAS_
    cudaStatus_ = cudaMalloc( (void**)&face_areas_,
			      number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_areas_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_areas_, grid.face_areas_,
			      number_of_cells_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_areas_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy face_areas_:\t" << face_areas_ << "\n";
    
    // FACE_CELLS_
    cudaStatus_ = cudaMalloc( (void**)&face_cells_,
			      2 * number_of_faces_ * sizeof(int));
    checkError_("cudaMalloc(face_cells_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_cells_, grid.face_cells_,
			      2 * number_of_faces_ * sizeof(int),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_cells_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy face_cells_:\t" << face_cells_ << "\n";
			    
    // FACE_NORMALS
    cudaStatus_ = cudaMalloc( (void**)&face_normals_,
			      dimensions_ * number_of_faces_ * sizeof(double));
    checkError_("cudaMalloc(face_normals_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    cudaStatus_ = cudaMemcpy( face_normals_, grid.face_normals_,
			      dimensions_ * number_of_faces_ * sizeof(double),
			      cudaMemcpyDeviceToDevice);
    checkError_("cudaMemcpy(face_normals_) in DeviceGrid::DeviceGrid(const DeviceGrid&)");
    std::cout << "\tCopy face_normals_:\t" << face_normals_ << "\n";
    
    std::cout << "Created DeviceGrid from copy constructor!\n";
}


// Destructor
DeviceGrid::~DeviceGrid() {
    std::cout << "Destructor - " << dimensions_ << "\n";
    std::cout << "\t\t" << cell_centroids_;
    if( cell_centroids_ != 0 ) {
	std::cout << "\tDel cell_centriods_\n";
	cudaStatus_ = cudaFree(cell_centroids_);
	checkError_("cudaFree(cell_centroids_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << cell_facepos_;
    if ( cell_facepos_ != 0 ) {
	std::cout << "\tDel cell_facepos_\n";
	cudaStatus_ = cudaFree(cell_facepos_);
	checkError_("cudaFree(cell_facepos_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << cell_faces_;
    if ( cell_faces_ != 0 ) {
	std::cout << "\tDel cell_faces\n";
	cudaStatus_ = cudaFree(cell_faces_);
	checkError_("cudaFree(cell_faces_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << cell_volumes_;
    if ( cell_volumes_ != 0 ) {
	std::cout << "\tDel cell_volumes_\n";
	cudaStatus_ = cudaFree(cell_volumes_);
	checkError_("cudaFree(cell_volumes_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << face_areas_;
    if ( face_areas_ != 0 ) {
	std::cout << "\tDel face_areas_\n";
	cudaStatus_ = cudaFree(face_areas_);
	checkError_("cudaFree(face_areas_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << face_cells_;
    if ( face_cells_ != 0 ) {
	std::cout << "\tDel face_cells_\n";
	cudaStatus_ = cudaFree(face_cells_);
	checkError_("cudaFree(face_cells_) in DeviceGrid::~DeviceGrid()");
    }
    std::cout << "\t\t" << face_normals_;
    if ( face_normals_ != 0 ) {
	std::cout << "\tDel face_normals_\n";
	cudaStatus_ = cudaFree(face_normals_);
	checkError_("cudaFree(face_normals_) in DeviceGrid::~DeviceGrid()");
    }
}


int DeviceGrid::test(int a) {
    dimensions_ = a;
    return 4;
}






// Check if for CUDA error and throw OPM exception if there is one.
void DeviceGrid::checkError_(const std::string& msg) const {
    //std::cout << "checking...\n";
    if ( cudaStatus_ != cudaSuccess ) {
	std::cout << "HELLO!!!!?\n";
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: " << cudaGetErrorString(cudaStatus_));
    }
        
}
