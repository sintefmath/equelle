
#ifndef EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED



#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
//#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"

#include <iostream>

namespace equelleCUDA {


    template <int codim>
    CollOfScalar DeviceGrid::operatorExtend(const CollOfScalar& in_data,
					    const CollOfIndices<codim>& from_set,
					    const CollOfIndices<codim>& to_set) const 
    {
	if ( to_set.isFull() ) {
	    return wrapDeviceGrid::extendToFull(in_data, 
						from_set.device_vector(),
						to_set.size());
	}
	else {
	    // Better safe than sorry:
	    int full_size;// = number_of_faces_;
	    if (codim == 0) { // cells
	        full_size = number_of_cells_;
	    } else if (codim == 1) {
	        full_size = number_of_faces_;
	    } else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << codim);
	    }
	    return wrapDeviceGrid::extendToSubset(in_data,
						  from_set.device_vector(),
						  to_set.device_vector(),
						  full_size);
	}
    
    }

    template <int codim>
    CollOfScalar DeviceGrid::operatorOn(const CollOfScalar& in_data,
					const CollOfIndices<codim>& from_set,
					const CollOfIndices<codim>& to_set) 
    {
	if ( from_set.isFull() ) {
	    return wrapDeviceGrid::onFromFull(in_data,
					      to_set.device_vector());
	    
	}
	else {
	    int full_size; // better safe than sorry
	    if (codim == 0) {
		full_size = number_of_cells_;
	    }
	    else if (codim == 1) {
		full_size = number_of_faces_;
	    }
	    else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << codim);
	    }
	    return wrapDeviceGrid::onFromSubset(in_data,
						from_set.device_vector(),
						to_set.device_vector(),
						full_size);
	}
    }



    template<int codim_data, int codim_set>
    thrust::device_vector<int> DeviceGrid::operatorOn( const CollOfIndices<codim_data>& in_data,
						      const CollOfIndices<codim_set>& from_set,
						      const CollOfIndices<codim_set>& to_set)
    {
	if ( from_set.isFull() ) {
	    return wrapDeviceGrid::onFromFullIndices(in_data.device_vector(),
						     to_set.device_vector());
	    
	}
	else {
	    int full_size; // better safe than sorry
	    if (codim_set == 0) {
		full_size = number_of_cells_;
	    }
	    else if (codim_set == 1) {
		full_size = number_of_faces_;
	    }
	    else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << codim_set);
	    }
	    return wrapDeviceGrid::onFromSubsetIndices(in_data.device_vector(),
						       from_set.device_vector(),
						       to_set.device_vector(),
						       full_size);
	}
    }



}  // namespace equelleCUDA



#endif // EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
