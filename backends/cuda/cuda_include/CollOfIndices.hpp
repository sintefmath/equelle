
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
      This is simply a minimally extended thrust::device_vector.
      It is used such that given a complete set (allCells(), allFaces()) no extra 
      indices are stored. We flag this by a boolean member variable.
      For subsets the indices within the set is stored as a thrust::device_vector.

      Since the class inherite thrust::device_vector, all vector member functions are
      available and public.
    */
    class CollOfIndices : public thrust::device_vector<int> // CollOfIndices
    {	
	// make device_vector private member. Implement only functions we need.
    public:
	//! Default constructor
	CollOfIndices();
	
	//! Constructor used for full sets.
	/*!
	  Insert value 'true' to create a full collection.
	  No extra storage used.
	  Will throw an exception if input value is 'false'.
	  \param full boolean indicating that the collection is complete.
	*/
	explicit CollOfIndices(const bool full);
	//! Constructor for creating a subset Coll
	/*!
	  Give as input the indices that will be part of the collection.
	*/
	explicit CollOfIndices(const thrust::device_vector<int>& indices);
	
	//! Copy constructor
	CollOfIndices(const CollOfIndices& coll);
	
	//! Destructor
	~CollOfIndices();
	
	/*! 
	  \return true if the Collection is full, false otherwise
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
	
    private:
	bool full_;
	
    }; // class CollOfIndices


} // namespace equelleCUDA


#endif // EQUELLE_COLLOFINDICES_HEADER_INCLUDED
