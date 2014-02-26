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

#include "deviceGrid.hpp"
#include "collOfScalar.hpp"
#include "CollOfIndices.hpp"

using namespace equelleCUDA;



// -------------------------------------------------- //
// ------- Implementation of CollOfIndices ---------- //
// -------------------------------------------------- //


CollOfIndices::CollOfIndices() 
    : thrust::device_vector<int>(),
      full_(false)
{
}


CollOfIndices::CollOfIndices(const bool full)
    : thrust::device_vector<int>(),
      full_(full)
{
    if (full_ != true ) {
	OPM_THROW(std::runtime_error, "Creating non-full CollOfIndices without giving the collection\n");
    }
}


CollOfIndices::CollOfIndices(const thrust::device_vector<int>& indices) 
    : thrust::device_vector<int>(indices.begin(), indices.end()),
      full_(false)
{
}


CollOfIndices::CollOfIndices(const CollOfIndices& coll)
    : thrust::device_vector<int>(coll.begin(), coll.end()),
    full_(coll.full_)
{
}

CollOfIndices::~CollOfIndices() 
{
    // The destructor do nothing. Automaticly calling base constructor.
}

bool CollOfIndices::isFull() const
{
    return full_;
}

thrust::host_vector<int> CollOfIndices::toHost() const {
    return thrust::host_vector<int>(this->begin(), this->end());
}



