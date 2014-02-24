
#ifndef EQUELLE_DEVICEGRID_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include <opm/core/grid/GridManager.hpp>


namespace equelleCUDA 
{

    	
    class DeviceGrid {

    public:
	DeviceGrid();
	explicit DeviceGrid( const UnstructuredGrid& grid);
	
	~DeviceGrid();

    private:
	
	// Member variables for unstructured grids
	const int dimensions_;
	const int number_of_cells_;
	const int number_of_faces_;
	
	// Member arrays for unstructured grids
	double* cell_centroids_; 
	int* cell_facepos_;
	int* cell_faces_;
	double* cell_volumes_;
	double* face_areas_;
	int* face_cells_;
	double* face_normals_;

	
	// Error handling:
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;

    }; // class DeviceGrid



} // namespace equelleCUDA

#endif // EQUELLE_DEVICEGRID_HEADER_INCLUDE 
