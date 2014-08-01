#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/merge.h>
#include <thrust/unique.h>
#include <thrust/for_each.h>

#include <opm/core/utility/ErrorMacros.hpp>

#include "wrapCollOfIndices.hpp"
#include "equelleTypedefs.hpp"

using namespace equelleCUDA;


void wrapCollOfIndices::containsFull(const thrust::device_vector<int>& subset,
				     const int full_size,
				     const int codim,
				     const std::string& name) {
    // subset is sorted.
    // Check only first and large element of subset.

    if (subset[0] < 0 ) {
	OPM_THROW(std::runtime_error, "Input set " << name << " contains invalid (negative) indices");
    }
    if ( subset[subset.size()-1] >= full_size ) {
	if (codim == 0) {
	    OPM_THROW(std::runtime_error, "Input set " << name << " contains indices larger than number_of_cells_ - 1 = " << full_size-1);
	}
	else {
	    OPM_THROW(std::runtime_error, "Input set " << name << " contains indices larger than number_of_faces_ - 1 = " << full_size-1);
	}
    }
}


void wrapCollOfIndices::containsSubset(const thrust::device_vector<int>& superset,
				       const thrust::device_vector<int>& subset,
				       const int codim,
				       const std::string& name) {

    //merging:
    thrust::device_vector<int> merged(superset.size() + subset.size());
    thrust::device_vector<int>::iterator merge_end = thrust::merge(superset.begin(), superset.end(), subset.begin(), subset.end(), merged.begin());
    // unique:
    thrust::device_vector<int>::iterator merge_new_end = thrust::unique(merged.begin(), merge_end);

    thrust::device_vector<int> hopefully_superset(merged.begin(), merge_new_end);
    if ( hopefully_superset.size() != superset.size()) {
	OPM_THROW( std::runtime_error, "Input set " << name << " is not a subset of the given set in the function call.");
    }
}


CollOfBool wrapCollOfIndices::isEmpty(const thrust::device_vector<int>& indices) {
    thrust::device_vector<int> temp(indices.begin(), indices.end());
    thrust::for_each(temp.begin(), temp.end(), functorIsEmpty());
    return CollOfBool(temp.begin(), temp.end());
}