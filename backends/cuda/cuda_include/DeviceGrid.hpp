
#ifndef EQUELLE_DEVICEGRID_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <opm/grid/GridManager.hpp>

#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "CollOfVector.hpp"


namespace equelleCUDA 
{
    
    // Forward declaration:
    //class CollOfScalar;

    //! Storing an unstructured grid on the device and do operations on this grid
    /*!
      The DeviceGrid class is based on the struct UnstructuredGrid from OPM, 
      and which is used by the current serial back-end for the Equelle compiler.
      Here, operations on the grid is given as member functions in DeviceGrid.

      The class contains the following data members as private variables for storing 
      the grid:
      - int dimensions_ 
          + Denoting 2D or 3D grid.
      - int number_of_cells_ 
          + Denoting the number of cells in the grid.
      - int number_of_faces_ 
          + Denoting the number og faces in the grid.
      - int size_cell_faces_
          + integer denoting the size of the cell_faces_ array. Might be removed on 
	  a later time.
	  
      - double* cell_centroids_
          + Points to array of coordinates of the centroids of all cells. Size of the 
	  array is ( dimensions_ * number_of_cells_ )
      - double* face_centroids_
          + Points to array of coordinates of the centroids of all faces. Size of the
	  array is ( dimensions_ * number_of_faces_ )
      - int* cell_facepos_
	  + Contains the range of indices for which faces belongs to each cell. 
	  The face indices are found in the array cell_faces_, and a cell c is 
	  surrounded by the faces from cell_faces_[cell_facepos_[c]] to 
	  (but not included) cell_faces_[cell_facepos_[c+1]]. Number of faces for 
	  cell c is therefore cell_facepos_[c+1] - cell_facepos_[c], and the size 
	  of cell_facepos_ is number_of_cells_ + 1.
      - int* cell_faces_
	  + Contains the indices for all faces belonging to each face as described above.
	  The size of the array is the value stored in cell_facepos_[number_of_cells_].
      - double* cell_volumes_
	  + Contains the volume of each cell. Size of array is number_of_cells_
      - double* face_areas_
	  + Contains the area of each face. Size of array is number_of_faces_
      - int* face_cells_
	  + Contains the indices of the two cells on each side of each face. 
	  Face f is the face seperating the cells face_cells_[2*f] and 
	  face_cells_[2*f + 1]. The orientation is also so that the normal of 
	  face f points from face_cells_[2*f] to face_cells_[2*f + 1]. If the face is 
	  on the boundary, then one of the two cells will have the value -1. 
	  Size of this array is number_of_faces * 2.
      - double* face_normals_
	  + Contains the normal vectors of each face, in non-normalized format.
	  In order to get the normalized normal vector, divide on the face_area_. 
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

	// Grid operations
	//! Collection of all cells
	/*!
	  \return CollOfIndices corresponding to all cells of the grid
	*/
	CollOfCell allCells() const;
	//! Collection of all faces
	/*!
	  \return CollOfIndices corresponding to all faces of the grid
	*/
	CollOfFace allFaces() const;

	//! Collection of all boundary faces
	/*!
	  Creates a collection of the indices of the boundary faces of the grid.
	  \return CollOfIndices of boundary faces
	*/
	const CollOfFace& boundaryFaces() const;
	//! Collection of all interior faces
	/*!
	  Creates a collection of the indices of the interior faces of the grid.
	  \return CollOfIndices of interior faces
	*/
	const CollOfFace& interiorFaces() const;
	
	//! Collection of all interior cells.
	/*!
	  Creates a collection of the indices of the interior cells of the grid.
	  \return CollOfIndices of interior cells.
	*/
	const CollOfCell& interiorCells() const;
	//! Collection of all boundary cells.
	/*!
	  Creates a collection of the indices of the boundary cells of the grid.
	  \return CollOfIndices of boundary cells.
	*/
	const CollOfCell& boundaryCells() const;

	//! Collection of first cell for a set of faces
	/*!
	  All faces have an orientation given by its normal vector, and each face
	  separates two cells (or one cell and the outside of the domain). 
	  The first and second cell are defined by this orientation, so that 
	  the face's normal vector points from FirstCell to SecondCell. 
	  This function gives the first cell of the set of faces given as input.
	  
	  If a face do not have a FirstCell, the output value for this face will be
	  -1. This will be the case for boundary faces with an inwards pointing
	  normal vector.

	  \param coll A collection of indices representing all faces, or a subset of 
	  all faces, in the grid.
	  \return A collection of indices representing the cells that are a FirstCell 
	  for the faces given as input. 

	  \sa secondCell
    	*/
	CollOfCell firstCell(CollOfFace coll) const;

	//! Collection of second cell for a set of faces
	/*!
	  All facs have an orientation given by its bormal vector, and each face
	  separates two cells (or one cell and the outside of the domain).
	  The first and second cell are defined by this orientation, so that the
	  face's normal vector points from FirstCell to SecondCell.
	  This function gives the second cell of the set of faces given as input.

	  If a face fo not have a SecondCell, the output value for this face will be
	  -1. This will be the case for boundary faces with an outwards pointing
	  normal vector.
	  
	  \param coll A collection of indices representing all faces, or a subset of
	  all faces, in the grid.
	  \return A collection of indices representing the cells taht are a SecondCell
	  for the faces given as input.
	  
	  \sa firstCell
	*/
	CollOfCell secondCell(CollOfFace coll) const;

	//! Implementation of the Equelle keyword Extend.
	/*!
	  The function takes as input a Collection of Scalars from one domain,
	  and map them over to a larger domain by expanding the Collection Of Scalars
	  by zeros. The input set has to be a subset of the output set.

	  \param[in] in_data Collection of Scalars that should be expanded.
	  \param[in] from_set The Collection of Indices corresponding
	  to in_data.
	  \param[in] to_set The Collection of Indices corresponding to the 
	  set given as return value.

	  \remark The sizes of in_data and from_set has to be the same,
	  and from_set has to be a subset of to_set.

	  \return Collection Of Scalars corresponding to to_set. Values for indices
	  found in from_set will be the same as in the corresponding position in
	  in_data, and the rest of the elements are zero.
	 */
	template<int codim>
	CollOfScalar operatorExtend(const CollOfScalar& in_data,
				    const CollOfIndices<codim>& from_set,
				    const CollOfIndices<codim>& to_set) const;

	//! Implementation of the Equelle keyword On for Collection of Scalars
	/*!
	  The function evaluate a evaluate-on or restrict-to operation of a 
	  Collection of Scalars defined on a Collection of Indices (Face or Cell)
	  and map them to a given subset of Indices of that.
	  
	  \param[in] in_data Collection of Scalars that should be restricted on
	  a subset set of what they already are defined for.
	  \param[in] from_set The indices on which in_data is defined.
	  \param[in] to_set A set of indices which the return collection should be
	  defined on. The to_set should be a subset of the in_data.
	  
	  \remark The sizes of in_data and from_set should be the same. 
	  
	  \return Collection Of Scalars corresponding to the to_set. Values are
	  the same for the ones in in_data for the corresponding indices in 
	  from_set and to_set.
	*/
	template<int codim>
	CollOfScalar operatorOn(const CollOfScalar& in_data,
				const CollOfIndices<codim>& from_set,
				const CollOfIndices<codim>& to_set);

	//! Implementation of the Equelle keyword On for CollOfIndices<codim>
	/*!
	  The function evaluate a evaluate-on or restrict-to operation of a 
	  Collection of Indices defined on a Collection of Indices (Face or Cell)
	  and map them to a given subset of Indices of that.
	  
	  \param[in] in_data Collection of Indices that should be restricted on
	  a subset set of what they already are defined for.
	  \param[in] from_set The indices on which in_data is defined.
	  \param[in] to_set A set of indices which the return collection should be
	  defined on. The to_set should be a subset of the in_data.
	  
	  \remark The sizes of in_data and from_set should be the same. The
	  collection given as in_data can be regarded as a Collection of Scalar
	  and this function will have the same functionality as the other operatorOn
	  function.
	  
	  \return Collection Of Indices corresponding to the to_set. Values are
	  the same for the ones in in_data for the corresponding indices in 
	  from_set and to_set.
	*/
	template<int codim_data, int codim_set>
	thrust::device_vector<int> operatorOn( const CollOfIndices<codim_data>& in_data,
					       const CollOfIndices<codim_set>& from_set,
					       const CollOfIndices<codim_set>& to_set);
	
	
	// NORM Functions
	/*!
	  Gives the sizes of the cells in the given set.
	  
	  \param[in] cells The indices of the cells for which we want the volume.
	  \param[in] full True means that we want the norm of all cells in the grid,
	  and false means that we only want a subset. If true we only call a cudaMemcpy,
	  otherwise we call a kernel function.

	  return A Collection of the cell volumes for the cells given by the input.
	*/
	CollOfScalar norm_of_cells(const thrust::device_vector<int>& cells,
				   const bool full) const;
	
	/*!
	  Gives the sizes of the faces in the given set.

	  \param[in] faces The indices of the faces for which we want the area.
	  \param[in] full True means that we want the norm of all faces in the grid,
	  and false means that we only want a subset. If true we only call a cudaMemcpy,
	  otherwise we call a kernel function.

	  return A Collection of the face area for the cells given by the input.
	*/
	CollOfScalar norm_of_faces(const thrust::device_vector<int>& faces,
				   const bool full) const ;


	//! Creates a Vector of Centroids
	/*!
	  Returns a vector with the centroids from the given set of either
	  cells (codim = 0) or faces (codim = 1). 
	  \param indices A vector of indices. Is empty if full is true
	  \param full Indicates if the set is a full set or not
	  \param codim 0 for cells and 1 for faces.
	*/
	CollOfVector centroid(const thrust::device_vector<int>& indices,
			      const bool full,
			      const int codim) const;


	//! Finds the normal vectors for the given faces.
	/*!
	  Returns a collection of Vectors for the faces given in the input
	  CollOfFace. 
	*/
	CollOfVector normal( const CollOfFace& faces) const;

	// ---------------- Get functions: -------------------------------

	/*!
	  \return dimensions_, the dimensions of the grid
	*/
	int dimensions() const;
	/*!
	  \return number_of_cells_, the number of cells in the entire grid.
	*/
	int number_of_cells() const;
	/*!
	  \return number_of_faces_, the number of faces in the entire grid.
	*/
	int number_of_faces() const;
	/*!
	  \return Pointer to cell_facepos_, array with number of faces for each cell
	*/
	int* cell_facepos() const;
	/*!
	  \return Pointer to cell_faces_, array with face indices surrounding each cell
	*/
	int* cell_faces() const;
	/*!
	  \return Pointer to face_cells_, array with cell index on each side of each face.
	*/
	int* face_cells() const;

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
	double* face_centroids_;
	int* cell_facepos_;
	int* cell_faces_;
	double* cell_volumes_;
	double* face_areas_;
	int* face_cells_;
	double* face_normals_;
    

	mutable CollOfFace boundary_faces_;
	mutable CollOfFace interior_faces_;
	mutable CollOfCell boundary_cells_;
	mutable CollOfCell interior_cells_;

	mutable bool boundaryFacesEmpty_;
	mutable bool interiorFacesEmpty_;
	mutable bool boundaryCellsEmpty_;
	mutable bool interiorCellsEmpty_;

	void createBoundaryFaces_() const;
	void createInteriorFaces_() const;
	void createBoundaryCells_() const;
	void createInteriorCells_() const;

	// Error handling:
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;

    }; // class DeviceGrid


    // ------------- END OF CLASS ------------------------------------------

    namespace wrapDeviceGrid {
	
	
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

    //! Kernel for computing first cell for a subset of faces.
    /*!
      Performs the same operation as firstCellKernel, but only computed on a subset
      of the faces. We therefore need to read an index as well in order to find the 
      correct cell.
      \code first[i] = face_cells[2*face_index[i]] \endcode
          
      \param[out] first Array of size number_of_faces where the resulting index
      of the first cell is to be stored.
      \param[in] number_of_faces Number of faces in the subset of the grid.
      \param[in] face_index The indices for the faces in the subset.
      \param[in] face_cells Complete storage of the two cells on each side of all faces.
      
      \sa DeviceGrid::firstCell, firstCellKernel
    */
    __global__ void firstCellSubsetKernel( int* first,
					   const int number_of_faces,
					   const int* face_index,
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

    //! Kernel for computing second cell for a subset of faces.
    /*!
      Performs the same operation as secondCellKernel, but only computed on a subset
      of the faces. We therefore need to read an index as well in order to find the
      correct cell.
      \code second[i] = face_cells[2*face_index[i] + 1] \endcode
                
      \param[out] second Array of size number_of_faces where the resulting index
      of the second cell is to be stored.
      \param[in] number_of_faces Number of faces in the subset of the grid.
      \param[in] face_index The indices for the faces in the subset.
      \param[in] face_cells Complete storage of the two cells on each side of all faces.
      
      \sa DeviceGrid::secondCell, secondCellKernel
    */
    __global__ void secondCellSubsetKernel( int* second,
					    const int number_of_faces,
					    const int* face_index,
					    const int* face_cells);

    // Kernel for finding the norm (natural sizes) of faces and cells
    /*!
      The sizes of faces and cells are stored in the DeviceGrid class, and this kernel 
      copies a given subset of these over to a collection of scalar memory location.
      
      \param[out] out Resulting norm which we want.
      \param[in] indices The subset of the grid for which we want to compute the sizes.
      \param[in] out_size Number of elements in out and indices.
      \param[in] norm_values A pointer to the array of precomputed values for face areas 
      or cell volumes, depending on which set we ask for.

      \sa DeviceGrid::norm_of_cells, DeviceGrid::norm_of_faces
    */
    __global__ void normKernel( double* out,
				const int* indices,
				const int out_size,
				const double* norm_values);


    //! Kernel for finding centroids from a subset of indices.
    /*!
      For finding the centroids of AllFaces or AllCells, we can simply perform
      a cudaMemcpy, since all centroid are already stored on the device. If we 
      however only want a subset of these centroid values, we need a kernel that
      reads only the required values.

      The centroids are stored as vectors with their coordinates. Currently this
      kernel is implemented as using one thread per vector, so that each thread 
      performs dimensions reads and writes.
      
      \param[out] out The resulting set of centroid vectors.
      \param[in] subset_indices The indices of the collection of faces or cells 
      that we want to store the centroids of.
      \param[in] all_centroids This is either the array cell_centroids_ or 
      face_centroids_ depending of what kind of centroids we are requesting.
      \param[in] num_vectors Number of vectors the result will be storing
      \param[in] dimensions Dimension of each vector.
    */
    __global__ void centroidKernel( double* out,
				    const int* subset_indices,
				    const double* all_centroids,
				    const int num_vectors,
				    const int dimensions);

    //! Kernel for finding the normal vectors of a subset of faces.
    /*!
      For finding the normal vectors of AllFaces we can simply make a cudaMemcpy call,
      but when we only want the normal vectors of a subset of AllFaces we call this 
      kernel to only read the ones we need.

      The kernel use one thread for each face, so that each thread copies dimension 
      values from all_face_normals to out.

      \param[out] out The resulting set of normal vectors.
      \param[in] faces The indices of the faces we want to find the normal vector of.
      \param[in] all_face_normals The array holding the normal vector for all faces
      in the grid.
      \param[in] num_vectors The number of faces in faces.
      \param[in] dimensions The dimension the grid is in, and therefore also the 
      dimension of each normal vector.
    */
    __global__ void faceNormalsKernel( double* out,
				       const int* faces,
				       const double* all_face_normals,
				       const int num_vectors,
				       const int dimensions);


    } // namespace wrapDeviceGrid


} // namespace equelleCUDA

// For implementation of template member functions:
#include "DeviceGrid_impl.hpp"


#endif // EQUELLE_DEVICEGRID_HEADER_INCLUDE 


