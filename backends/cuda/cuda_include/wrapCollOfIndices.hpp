
#ifndef EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
#define EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#include <thrust/device_vector.h>

#include "equelleTypedefs.hpp"

namespace equelleCUDA 
{

    namespace wrapCollOfIndices 
    {

	void containsFull( const thrust::device_vector<int>& subset,
			   const int& full_size,
			   const int& codim,
			   const std::string& name);

	void containsSubset(const thrust::device_vector<int>& superset, 
			    const thrust::device_vector<int>& subset,
			    const int& codim,
			    const std::string& name);

	CollOfBool isEmpty(const thrust::device_vector<int>& indices);
	
	struct functorIsEmpty {
	    __host__ __device__
	    void operator()(int& x) {
		x= (x == -1);
	    }
	};

	
    } // namespace wrapCollOfIndices

} // namespace equelleCUDA


#endif // EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
