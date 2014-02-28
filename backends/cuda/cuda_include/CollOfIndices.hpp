
#ifndef EQUELLE_COLLOFINDICES_HEADER_INCLUDED
#define EQUELLE_COLLOFINDICES_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace equelleCUDA 
{

    //! Class for keeping track of subsets by storing a collection of indices.
    /*!
      This class is a tailored thrust::device_vector<int> class for storing indices
      for handeling subsets of faces and cells. The vector should store indices 
      of the cells or faces that is part of the set. 
      If we have a complete set (allCells(), allFaces()) we simply 
      set a flag to show it is full instead of storing N indices {0, 1, ..., N-1}.

      Vector functions for begin and end iterators, as well as getting a raw int pointer
      to the data is provided.
    */
    template <int dummy>
    class CollOfIndices 
    {	
    public:
	//! Default constructor
	CollOfIndices();
	
	//! Constructor used for full sets.
	/*!
	  Insert the size of the full collection
	  No extra storage used, sets a flag to show that it is full.
	*/
	explicit CollOfIndices(const int size);

	//! Constructor for creating a subset Coll
	/*!
	  Give as input the indices that will be part of the collection.
	*/
	explicit CollOfIndices(const thrust::device_vector<int>& indices);

	//! Constructor for creating a subset collection from an iterator range
	/*!
	  \param[in] begin A pointer to the first element of the range which should
	  be in the collection.
	  \param[in] end A pointer to the element after the last one that should be
	  int the collection.
	 */
	explicit CollOfIndices(thrust::device_vector<int>::iterator begin,
			       thrust::device_vector<int>::iterator end);
	
	//! Copy constructor
	CollOfIndices(const CollOfIndices& coll);
	
	//! Destructor
	~CollOfIndices();
	
	/*! 
	  \return True if the Collection is full, false otherwise
	*/
	bool isFull() const;
	//! Copy the device_vector to host memory.
	/*!
	  A member function to copy the values over to device memory.
	  This function is added since we not are able to cast Collection 
	  straight to thrust::host_vector. 
	  Most naturally used in debugging.
	*/
	thrust::host_vector<int> toHost() const;
	
	//! Collection size
	/*!
	  \return The size of the collection regardless if it is full or not.
	*/
	int size() const;
	
	//! Begin iterator
	/*!
	  \return An iterator at the beginning of the device vector
	*/
	//thrust::device_vector<int>::iterator begin() const;
	thrust::device_vector<int>::iterator begin();

	//! End iterator
	/*!
	  \return An iterator at the end of the device vector.
	*/
	//	thrust::device_vector<int>::iterator end() const;
	thrust::device_vector<int>::iterator end();
	
	//! Reference to the first vector element
	/*!
	  \return An int pointer to the first element of the device_vector.
	  Useful when the vector data is needed in a kernel.
	*/
	int* raw_pointer();
	
    private:
	bool full_;
	int size_;
	thrust::device_vector<int> dev_vec_;
	
    }; // class CollOfIndices


    typedef CollOfIndices<0> CollOfCell;
    typedef CollOfIndices<1> CollOfFace;


} // namespace equelleCUDA

// Include implementation of the template class
#include "CollOfIndices_impl.hpp"


#endif // EQUELLE_COLLOFINDICES_HEADER_INCLUDED
