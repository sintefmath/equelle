
#ifndef EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
#define EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#include <thrust/device_vector.h>

#include "equelleTypedefs.hpp"

namespace equelleCUDA 
{
    //! Namespace for functions belonging to the class CollOfIndices, but which cannot be members.
    namespace wrapCollOfIndices 
    {
	
	//! Makes sure that the given subset is a valid subset of the full grid.
	/*!
	  When reading a domain from file we need to make sure that this is 
	  a valid subset. This function perform this test when we read a subset of 
	  AllFaces() or AllCells(). If the subset is valid, nothing happens.
	  If the subset is not valid the function throw an exception.

	  \param[in] subset The indices read from file.
	  \param[in] full_size Number of cells or faces depending on what we want a 
	  subset of.
	  \param[in] codim Co-dimension. 0 for cells, 1 for faces.
	  \param[in] name Name of the variable we try to read.
	*/
	void containsFull( const thrust::device_vector<int>& subset,
			   const int full_size,
			   const int codim,
			   const std::string& name);

	//! Makes sure that the given subset is a valid subset of another set of indices.
	/*!
	  When reading a domain from file we need to make sure that this is 
	  a valid subset of what we want. This function perform this test when 
	  we read a subset of something else than
	  AllFaces() or AllCells(). If the subset is valid, nothing happens.
	  If the subset is not valid the function throw an exception.

	  \param[in] superset The set of indices the input should be a subset of.
	  \param[in] subset The indices read from file.
	  \param[in] codim Co-dimension. 0 for cells, 1 for faces.
	  \param[in] name Name of the variable we try to read.

	*/
	void containsSubset(const thrust::device_vector<int>& superset, 
			    const thrust::device_vector<int>& subset,
			    const int codim,
			    const std::string& name);

	//! Check for empty cell / empty face flag in the input.
	/*!
	  Most often used for finding the interior cell of boundary faces.
	  First and Second Cell returns value -1 when there are no cell at 
	  those positions. This function check a set of indices and returns 
	  a Collection of Booleans which indicates if each index is empty or
	  not.
	*/
	CollOfBool isEmpty(const thrust::device_vector<int>& indices);
	
	//! Functor for checking if a cell is empty.
	/*!
	  Used with algorithms iterating through vectors. Given example:
	  \code 
	  thrust::device_vector<int> out = indices
	  thrust::for_each(out.begin(), out.end(), functorIsEmpty());
	  \endcode
	  Here the functor set each value x found in out to the boolean
	  value of (x == -1).
	*/
	struct functorIsEmpty {
	    __host__ __device__
	    /*!
	      x is given value 1 if x is -1, otherwise x is given value 0.
	     
	      Implementation:
	      \code x = (x == -1); \endcode
	    */
	    void operator()(int& x) {
		x= (x == -1);
	    }
	};

	
    } // namespace wrapCollOfIndices

} // namespace equelleCUDA


#endif // EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
