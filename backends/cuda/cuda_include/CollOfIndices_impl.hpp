
//#ifndef EQUELLE_COLLOFINDICES_IMPL_INCLUDED
//#define EQUELLE_COLLOFINDICES_IMPL_INCLUDED

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

using namespace equelleCUDA;



// -------------------------------------------------- //
// ------- Implementation of CollOfIndices ---------- //
// -------------------------------------------------- //

template <int dummy>
CollOfIndices<dummy>::CollOfIndices() 
    : full_(false),
      size_(0),
      dev_vec_(0)
{
}

template <int dummy>
CollOfIndices<dummy>::CollOfIndices(const int size)
    : full_(true),
      size_(size),
      dev_vec_(0)
{
    if (full_ != true ) {
	OPM_THROW(std::runtime_error, "Creating non-full CollOfIndices without giving the collection\n");
    }
}

template <int dummy>
CollOfIndices<dummy>::CollOfIndices(const thrust::device_vector<int>& indices) 
    : full_(false),
      size_(0),
      dev_vec_(indices.begin(), indices.end())
{
    size_ = dev_vec_.size();
}

template <int dummy>
CollOfIndices<dummy>::CollOfIndices(thrust::device_vector<int>::iterator begin,
			     thrust::device_vector<int>::iterator end)
    : full_(false),
      size_(0),
      dev_vec_(begin, end)
{
    size_ = dev_vec_.size();
}

template <int dummy>
CollOfIndices<dummy>::CollOfIndices(const CollOfIndices& coll)
    : full_(coll.full_),
      size_(coll.size_),
      dev_vec_(coll.dev_vec_.begin(), coll.dev_vec_.end())
{
}

template <int dummy>
CollOfIndices<dummy>::~CollOfIndices() 
{
    // Nothing we manually have to destruct.
}

template <int dummy>
bool CollOfIndices<dummy>::isFull() const
{
    return full_;
}

template <int dummy>
thrust::host_vector<int> CollOfIndices<dummy>::toHost() const {
    return thrust::host_vector<int>(dev_vec_.begin(), dev_vec_.end());
}

template <int dummy>
std::vector<int> CollOfIndices<dummy>::stdToHost() const {
    thrust::host_vector<int> host(dev_vec_.begin(), dev_vec_.end());
    return std::vector<int>(host.begin(), host.end());
}


template <int dummy>
int CollOfIndices<dummy>::size() const {
    return size_;
}

//thrust::device_vector<int>::iterator CollOfIndices::begin() const {
//    return dev_vec_.begin();
//}

//thrust::device_vector<int>::iterator CollOfIndices::end() const {
//    return dev_vec_.end();
//}


template <int dummy>
thrust::device_vector<int>::iterator CollOfIndices<dummy>::begin() {
    return dev_vec_.begin();
}

template <int dummy>
thrust::device_vector<int>::iterator CollOfIndices<dummy>::end() {
    return dev_vec_.end();
}


// This one should be const, but raw_pointer_cast is incompitible with const...
template <int dummy>
int* CollOfIndices<dummy>::raw_pointer() {
    //thrust::device_vector<int> temp(8);
    //thrust::fill(temp.begin(), temp.end(), 9);
    //const int* out = thrust::raw_pointer_cast( &dev_vec_[0] );
    //int* out2 = out;
    return thrust::raw_pointer_cast( &dev_vec_[0] );
}

/*
template <int dummy>
void CollOfIndices<dummy>::contains( CollOfIndices<dummy> subset,
				     const std::string& name) {
    
    // If this is a full set we only have to check first and last element
    if ( this->isFull() ) {
	int* dev_ptr = this->raw_pointer();
	if ( (dev_ptr[0] < 0) ) {
	    OPM_THROW(std::runtime_error, "Input set " << name << " contains invalid (negative) indices");
	}
	if (dev_ptr[subset.size()-1] >= this->size())  {
	    if ( dummy == 0) {
		OPM_THROW(std::runtime_error, "Input set " << name << " contains indices larger than number_of_cells " << this->size()); 
	    }
	    if (dummy == 1) {
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

template<int dummy>
void CollOfIndices<dummy>::sort() {
    thrust::sort(dev_vec_.begin(), dev_vec_.end());
}

*/
//template class CollOfIndices<0>;
//template class CollOfIndices<1>;


//#endif // EQUELLE_COLLOFINDICES_IMPL_INCLUDED
