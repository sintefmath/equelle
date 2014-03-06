
#ifndef EQUELLE_COLLOFINDICES_IMPL_INCLUDED
#define EQUELLE_COLLOFINDICES_IMPL_INCLUDED

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
#include <thrust/fill.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/remove.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/retag.h>
#include <thrust/sort.h>
#include <thrust/merge.h>
#include <thrust/unique.h>

//#include "DeviceGrid.hpp"
//#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "wrapCollOfIndices.hpp"

using namespace equelleCUDA;



// -------------------------------------------------- //
// ------- Implementation of CollOfIndices ---------- //
// -------------------------------------------------- //

template <int codim>
CollOfIndices<codim>::CollOfIndices() 
    : full_(false),
      size_(0),
      dev_vec_(0)
{
}

template <int codim>
CollOfIndices<codim>::CollOfIndices(const int size)
    : full_(true),
      size_(size),
      dev_vec_(0)
{
    if (full_ != true ) {
	OPM_THROW(std::runtime_error, "Creating non-full CollOfIndices without giving the collection\n");
    }
}

template <int codim>
CollOfIndices<codim>::CollOfIndices(const thrust::device_vector<int>& indices) 
    : full_(false),
      size_(0),
      dev_vec_(indices.begin(), indices.end())
{
    size_ = dev_vec_.size();
}

template <int codim>
CollOfIndices<codim>::CollOfIndices(thrust::device_vector<int>::iterator begin,
			     thrust::device_vector<int>::iterator end)
    : full_(false),
      size_(0),
      dev_vec_(begin, end)
{
    size_ = dev_vec_.size();
}

template <int codim>
CollOfIndices<codim>::CollOfIndices(const CollOfIndices& coll)
    : full_(coll.full_),
      size_(coll.size_),
      dev_vec_(coll.dev_vec_.begin(), coll.dev_vec_.end())
{
}

template <int codim>
CollOfIndices<codim>::~CollOfIndices() 
{
    // Nothing we manually have to destruct.
}

template <int codim>
bool CollOfIndices<codim>::isFull() const
{
    return full_;
}

template <int codim>
thrust::host_vector<int> CollOfIndices<codim>::toHost() const {
    return thrust::host_vector<int>(dev_vec_.begin(), dev_vec_.end());
}

template <int codim>
thrust::device_vector<int> CollOfIndices<codim>::device_vector() const {
    return dev_vec_;
}

template <int codim>
std::vector<int> CollOfIndices<codim>::stdToHost() const {
    thrust::host_vector<int> host(dev_vec_.begin(), dev_vec_.end());
    return std::vector<int>(host.begin(), host.end());
}


template <int codim>
int CollOfIndices<codim>::size() const {
    return size_;
}

//thrust::device_vector<int>::iterator CollOfIndices::begin() const {
//    return dev_vec_.begin();
//}

//thrust::device_vector<int>::iterator CollOfIndices::end() const {
//    return dev_vec_.end();
//}


template <int codim>
thrust::device_vector<int>::iterator CollOfIndices<codim>::begin() {
    return dev_vec_.begin();
}

template <int codim>
thrust::device_vector<int>::iterator CollOfIndices<codim>::end() {
    return dev_vec_.end();
}


// This one should be const, but raw_pointer_cast is incompitible with const...
template <int codim>
int* CollOfIndices<codim>::raw_pointer() {
    //thrust::device_vector<int> temp(8);
    //thrust::fill(temp.begin(), temp.end(), 9);
    //const int* out = thrust::raw_pointer_cast( &dev_vec_[0] );
    //int* out2 = out;
    return thrust::raw_pointer_cast( &dev_vec_[0] );
}


template <int codim>
void CollOfIndices<codim>::contains( CollOfIndices<codim> subset,
				     const std::string& name) {
 
    if ( this->isFull() ) {
	// Check first and last element
	// Throws exception if subset is not contained.
	wrapCollOfIndices::containsFull(subset.device_vector(), this->size(),
					codim, name);
     }
    else {
	// Need to compare two CollOfIndices vectors:
	// Throws an exception if subset is not contained.
	wrapCollOfIndices::containsSubset(this->device_vector(),
					  subset.device_vector(),
					  codim, name);
    }
}
    /*    
    // If this is a full set we only have to check first and last element
    if ( this->isFull() ) {
	int* dev_ptr = this->raw_pointer();
	if ( (dev_ptr[0] < 0) ) {
	    OPM_THROW(std::runtime_error, "Input set " << name << " contains invalid (negative) indices");
	}
	if (dev_ptr[subset.size()-1] >= this->size())  {
	    if ( codim == 0) {
		OPM_THROW(std::runtime_error, "Input set " << name << " contains indices larger than number_of_cells " << this->size()); 
	    }
	    if (codim == 1) {
		OPM_THROW(std::runtime_error, "Input set " << name << " contains indices large than number_of_faces " << this->size());
	    }	
	}
    }
    
    // else the superset is not full
    else {
	// merging:
	thrust::device_vector<int> merged(this->size() + subset.size());
	thrust::device_vector<int>::iterator merge_end = thrust::merge(this->begin(), this->end(), subset.begin(), subset.end(), merged.begin());

	// unique:
	thrust::device_vector<int>::iterator merge_new_end = thrust::unique(merged.begin(), merge_end);

	thrust::device_vector<int> hopefully_superset(merged.begin(), merge_new_end);
	if ( hopefully_superset.size() != this->size()) {
	    OPM_THROW(std::runtime_error, "Input set " << name << " is not a subset by the given set in the function call.");
	}
    }
}
    */

template<int codim>
void CollOfIndices<codim>::sort() {
    // thrust::sort(dev_vec_.begin(), dev_vec_.end());
    //wrapCollOfIndices::sort(begin(), end());
}


//template class CollOfIndices<0>;
//template class CollOfIndices<1>;


#endif // EQUELLE_COLLOFINDICES_IMPL_INCLUDED
