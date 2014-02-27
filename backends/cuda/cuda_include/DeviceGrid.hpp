
#ifndef EQUELLE_DEVICEGRID_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <opm/core/grid/GridManager.hpp>

#include "CollOfIndices.hpp"

namespace equelleCUDA 
{
    	
    //! Storing an unstructured grid on the device and do operations on this grid
    /*!
      The DeviceGrid class is based on the struct UnstructuredGrid from OPM, and which is used by the current serial back-end for the Equelle compiler. Here, operations on the grid is given as member functions in DeviceGrid.

      The class contains the following data members as private variables for storing the grid:
      - int dimensions_ 
          + Denoting 2D or 3D grid.
      - int number_of_cells_ 
          + Denoting the number of cells in the grid.
      - int number_of_faces_ 
          + Denoting the number og faces in the grid.
      - int size_cell_faces_
          + integer denoting the size of the cell_faces_ array. Might be removed on a later time.
	  
      - double* cell_centroids_
          + Points to array of coordinates of the centroids of all cells. Size of the array is ( dimensions_ * number_of_cells_ )
      - int* cell_facepos_
	  + Contains the range of indices for which faces belongs to each cell. The face indices are found in the array cell_faces_, and a cell c is surrounded by the faces from cell_faces_[cell_facepos_[c]] to (but not included) cell_faces_[cell_facepos_[c+1]]. Number of faces for cell c is therefore cell_facepos_[c+1] - cell_facepos_[c], and the size of cell_facepos_ is number_of_cells_ + 1.
      - int* cell_faces_
	  + Contains the indices for all faces belonging to each face as described above. The size of the array is the value stored in cell_facepos_[number_of_cells_].
      - double* cell_volumes_
	  + Contains the volume of each cell. Size of array is number_of_cells_
      - double* face_areas_
	  + Contains the area of each face. Size of array is number_of_faces_
      - int* face_cells_
	  + Contains the indices of the two cells on each side of each face. Face f is the face seperating the cells face_cells_[2*f] and face_cells_[2*f + 1]. The orientation is also so that the normal of face f points from face_cells_[2*f] to face_cells_[2*f + 1]. If the face is on the boundary, then one of the two cells will have the value -1. Size of this array is number_of_faces * 2.
      - double* face_normals_
	  + Contains the normal vectors of each face, in non-normalized format. In order to get the normalized normal vector, divide on the face_area_. 
     */
    class DeviceGrid {
    public:
	/*! 
	  Default constructor. Sets all members to zero.
	*/
	DeviceGrid();
	
	//! Constructor from UnstructuredGrid
	/*! 
	  The only way to initialize a the grid on the device.

	  The arrays in the DeviceGrid class is a subset of the members of the struct
	  UnstructuredGrid, and the constructor allocates memory and copies
	  the arrays over to the device. 
	*/
	explicit DeviceGrid( const UnstructuredGrid& grid);

	//! Copy constructor
	/*!
	  Allocates memory for a new grid and uses a device to device copy.
	*/
	DeviceGrid( const DeviceGrid& grid);

	// This is a pure debugger function.
	// We mark each collection with an ID to keep track of
	// which ones are destructed.
	//! Dummy function for debugging. REMOVE IN THE FUTURE!
	int setID(int a);

	// Grid operations
	//! Collection of all cells
	/*!
	  \return CollOfIndices corresponding to all cells of the grid
	*/
	CollOfIndices allCells() const;
	//! Collection of all faces
	/*!
	  \return CollOfIndices corresponding to all faces of the grid
	*/
	CollOfIndices allFaces() const;

	//! Collection of all boundary faces
	/*!
	  Creates a collection of the indices of the boundary faces of the grid.
	  \return CollOfIndices of boundary faces
	*/
	CollOfIndices boundaryFaces() const;
	//! Collection of all interior faces
	/*!
	  Creates a collection of the indices of the interior faces of the grid.
	  \return CollOfIndices of interior faces
	*/
	CollOfIndices interiorFaces() const;
	
	//! Collection of all interior cells.
	/*!
	  Creates a collection of the indices of the interior cells of the grid.
	  \return CollOfIndices of interior cells.
	*/
	CollOfIndices interiorCells() const;
	//! Collection of all boundary cells.
	/*!
	  Creates a collection of the indices of the boundary cells of the grid.
	  \return CollOfIndices of boundary cells.
	*/
	CollOfIndices boundaryCells() const;

	//! Collection of first cell for all faces
	/*!
	  All faces have an orientation given by its normal vector, and each face
	  separates two cells (or one cell and the outside of the domain). 
	  The first and second cell are defined by this orientation, so that 
	  the face's normal vector points from FirstCell to SecondCell.

	  The collection here contains number_of_faces_ entries of indices
	  of the FirstCell of each face respectively. If a face is on the boundary
	  with an inwards pointing normal vector, it has no FirstCell, and this is
	  represented by a -1 value in the collection.
	  
	  \sa secondCell
    	*/
	CollOfIndices firstCell() const;

	//! Collection of second cell for all faces
	/*!
	  All facs have an orientation given by its bormal vector, and each face
	  separates two cells (or one cell and the outside of the domain).
	  The first and second cell are defined by this orientation, so that the
	  face's normal vector points from FirstCell to SecondCell.
	  
	  The collection here contains number_of_faces_ entries of indices 
	  of the SecondCell of each face respectively. If a face is on the boundary 
	  with an outwards pointing normal vector, it has no SecondCell, and this is
	  represented by a -1 value in the collection.
	  
	  \sa firstCell
	*/
	CollOfIndices secondCell() const;


	// Get functions:
	/*!
	  \return dimensions
	*/
	int dimensions() const;
	/*!
	  \return number of cells in the entire grid.
	*/
	int number_of_cells() const;
	/*!
	  \return number of faces in the entire grid.
	*/
	int number_of_faces() const;
	
	//! Destructor
	~DeviceGrid();

    private:

	// Member variables for unstructured grids
	const int dimensions_;
	const int number_of_cells_;
	const int number_of_faces_;
	const int size_cell_faces_; // Extra variable compared to UnstructuredGrid
	
	// Member arrays for unstructured grids
	double* cell_centroids_; 
	int* cell_facepos_;
	int* cell_faces_;
	double* cell_volumes_;
	double* face_areas_;
	int* face_cells_;
	double* face_normals_;
	

	int id_;
	
	// Error handling:
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;

    }; // class DeviceGrid


    //! Functor checking if an input is equal to its stored value.
    /*! 
      This functor is used together with iterators for evaluating a function to all
      elements of a vector.

      The functor is initialized with a referance value. Then each iterated element
      will be compared to the this value, return true if they are equal and false if not.

      Example usage:
      \code
      // Create vector of N elements all with value -1
      vector<int> vec(N, -1);
      // modify some of the values:
      modify(vec);
      // remove the elements which is not modified:
      iterator new_end = remove_if(vec.begin(), vec.end(), unchanged(-1));
      \endcode
     */
    struct unchanged
    {
	//! Value all input will be compared with.
	const int val;
	//! Constructor
	/*!
	  Sets the value which all input will be compared with
	*/
	unchanged(int val_in) : val(val_in) {}
	//! The comparison function.
	/*!
	  Can be called from either host or device code.
	  \param x from input
	  \return \code (x == val) \endcode
	 */
	__host__ __device__ 
	bool operator()(const int x) {
	    return (x == val); 
	}
    };


    // KERNELS

    //! Kernel for computing boundary faces in a grid.
    /*!
      Computing if each face is a boundary face or not. 
      \param[in,out] b_faces Array with number_of_faces elements. All elements should be initialized to hold a number representing an illegal index. Then each face modifies its corresponding element to hold its index if it is a boundary face.
      \param[in] face_cells Array containing cell indices for each side of every face. Contains -1 if the face is on the boundary. Array size: number_of_faces * 2.
      \param[in] number_of_faces Number of faces in the grid.
      
      \sa DeviceGrid::boundaryFaces
    */
    __global__ void boundaryFacesKernel( int* b_faces,
					 const int* face_cells,
					 const int number_of_faces);

    //! Kernel for computing interior faces in a grid
    /*!
      Computing if each face is an interior face or not.
      \param[in,out] i_faces Array with number_of_faces elements. All elements should be initialized to hold a number representing an illegal index. Then each face modifies its corresponding element to hold its index if it is an interior face.
      \param[in] face_cells Array containing cell indices for each side of every face. Contains -1 if the face is on the boundary. Array size: number_of_faces * 2.
      \param[in] number_of_faces Number of faces in the grid.
      
      \sa DeviceGrid::interiorFaces
     */
    __global__ void interiorFacesKernel( int* i_faces,
					 const int* face_cells,
					 const int number_of_faces);

    //! Kernel for computing boundary cells in a grid.
    /*!
      Computing if each cell is a boundary cell or not.
      \param[in,out] b_cells Array with number_of_cells elements. All elements should be initialized to hold a number representing an illegal index. Then each face midifies its corresponding element to hold its index if it is a boundary cell.
      \param[in] number_of_cells Number of cells in the grid.
      \param[in] cell_facepos Array of size number_of_cells + 1, containing the index range for each cell for reading faces in cell_faces. 
      \param[in] cell_faces Array with face indices surrounding all cells. Which face indices corresponding to which cell is given from cell_facepos.
      \param[in] face_cells Array containing cell indices for each side of every face. Contains -1 if the face is on the boundary. Array size: number_of_faces * 2.
      
      \sa DeviceGrid::boundaryCells
     */
    __global__ void boundaryCellsKernel( int* b_cells, // size number_of_cells
					 const int number_of_cells,
					 const int* cell_facepos, // number_of_cells + 1
					 const int* cell_faces, // size_cell_faces_
					 const int* face_cells); // 2 * number_of_faces
    
    //! Kernel for computing interior cells in a grid.
    /*!
      Computing if each cell is an interior cell or not.
      \param[in,out] i_cells Array with number_of_cells elements. All elements should be initialized to hold a number representing an illegal index. Then each face midifies its corresponding element to hold its index if it is an interior cell.
      \param[in] number_of_cells Number of cells in the grid.
      \param[in] cell_facepos Array of size number_of_cells + 1, containing the index range for each cell for reading faces in cell_faces. 
      \param[in] cell_faces Array with face indices surrounding all cells. Which face indices corresponding to which cell is given from cell_facepos.
      \param[in] face_cells Array containing cell indices for each side of every face. Contains -1 if the face is on the boundary. Array size: number_of_faces * 2.
      
      \sa DeviceGrid::interiorCells
    */
    __global__ void interiorCellsKernel( int* i_cells,
					 const int number_of_cells,
					 const int* cell_facepos,
					 const int* cell_faces,
					 const int* face_cells);

    //! Kernel for computing first cell for all faces.
    /*!
      The first cell for face f is simply found by
      \code first[f] = face_cells[2*f] \endcode
      
      \param[out] first Array of size number_of_faces where the resulting index
      of the first cell is to be stored.
      \param[in] number_of_faces Number of faces in the grid
      \param[in] face_cells Complete storage of the two cells on each side of all faces.
      
      \sa DeviceGrid::firstCell
    */
    __global__ void firstCellKernel( int* first,
				     const int number_of_faces,
				     const int* face_cells);

    //! Kernel for computing second cell for all faces
    /*!
      The second cell for face f is simply found by
      \code second[f] = face_cells[2*f + 1] \endcode
      
      \param[out] second Array of size number_of_faces where the resulting index
      of the second cell is to be stored.
      \param[in] number_of_faces Number of faces in the grid.
      \param[in] face_cells Complete storage of the two cells on each side of all faces.
      
      \sa DeviceGrid::secondCell
    */
    __global__ void secondCellKernel( int* second,
				      const int number_of_faces,
				      const int* face_cells);

} // namespace equelleCUDA

#endif // EQUELLE_DEVICEGRID_HEADER_INCLUDE 


