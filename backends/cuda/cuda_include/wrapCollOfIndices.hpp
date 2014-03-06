
#ifndef EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
#define EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <string>

#include <thrust/device_vector.h>


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


    
    } // namespace wrapCollOfIndices

} // namespace equelleCUDA


#endif // EQUELLE_WRAP_COLLOFINDICES_HEADER_INCLUDED
