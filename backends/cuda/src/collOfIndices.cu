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

#include "DeviceGrid.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"

using namespace equelleCUDA;



// -------------------------------------------------- //
// ------- Implementation of CollOfIndices ---------- //
// -------------------------------------------------- //


CollOfIndices::CollOfIndices() 
    : full_(false),
      size_(0),
      dev_vec_(0)
{
}


CollOfIndices::CollOfIndices(const int size)
    : full_(true),
      size_(size),
      dev_vec_(0)
{
    if (full_ != true ) {
	OPM_THROW(std::runtime_error, "Creating non-full CollOfIndices without giving the collection\n");
    }
}


CollOfIndices::CollOfIndices(const thrust::device_vector<int>& indices) 
    : full_(false),
      size_(0),
      dev_vec_(indices.begin(), indices.end())
{
    size_ = dev_vec_.size();
}

CollOfIndices::CollOfIndices(thrust::device_vector<int>::iterator begin,
			     thrust::device_vector<int>::iterator end)
    : full_(false),
      size_(0),
      dev_vec_(begin, end)
{
    size_ = dev_vec_.size();
}

CollOfIndices::CollOfIndices(const CollOfIndices& coll)
    : full_(coll.full_),
      size_(coll.size_),
      dev_vec_(coll.dev_vec_.begin(), coll.dev_vec_.end())
{
}

CollOfIndices::~CollOfIndices() 
{
    // Nothing we manually have to destruct.
}

bool CollOfIndices::isFull() const
{
    return full_;
}

thrust::host_vector<int> CollOfIndices::toHost() const {
    return thrust::host_vector<int>(dev_vec_.begin(), dev_vec_.end());
}


int CollOfIndices::size() const {
    return size_;
}

//thrust::device_vector<int>::iterator CollOfIndices::begin() const {
//    return dev_vec_.begin();
//}

//thrust::device_vector<int>::iterator CollOfIndices::end() const {
//    return dev_vec_.end();
//}

thrust::device_vector<int>::iterator CollOfIndices::begin() {
    return dev_vec_.begin();
}

thrust::device_vector<int>::iterator CollOfIndices::end() {
    return dev_vec_.end();
}

int* CollOfIndices::raw_pointer() {
    //thrust::device_vector<int> temp(8);
    //thrust::fill(temp.begin(), temp.end(), 9);
    //int* out = thrust::raw_pointer_cast( &temp[0] );
    //int* out2 = out;
    return thrust::raw_pointer_cast( &dev_vec_[0] );
}