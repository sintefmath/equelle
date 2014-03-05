
#ifndef EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED



#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
//#include "DeviceGrid.hpp"

#include <iostream>

namespace equelleCUDA {


    template <int dummy>
    CollOfScalar DeviceGrid::operatorExtend(const CollOfScalar& in_data,
					    const CollOfIndices<dummy>& from_set,
					    const CollOfIndices<dummy>& to_set) 
    {
	if ( to_set.isFull() ) {
	    std::cout << "EXTEND\n";
	    return wrapDeviceGrid::extendToFull(in_data, 
						from_set.device_vector(),
						to_set.size());
	}
	else {
	    // Better safe than sorry:
	    int full_size = number_of_faces_;
	    //if (dummy == 0) { // cells
	    //    full_size = number_of_cells_;
	    //} else (dummy == 1) {
	    //    full_size = number_of_faces_;
	    //}
	    return wrapDeviceGrid::extendToSubset(in_data,
						  from_set.device_vector(),
						  to_set.device_vector(),
						  full_size);
	    //OPM_THROW(std::runtime_error, "Extend from subset to subset of full is not yet implemented");
	}
    
    }

    template <int dummy>
    CollOfScalar DeviceGrid::operatorOn(const CollOfScalar& in_data,
					const CollOfIndices<dummy>& from_set,
					const CollOfIndices<dummy>& to_set) 
    {
	std::cout << "\n\nON\n\n";
	if ( from_set.isFull() ) {
	    return wrapDeviceGrid::onFromFull(in_data,
					      to_set.device_vector());
	    
	}
	else {
	    int full_size = number_of_faces_; // better safe than sorry
	    return wrapDeviceGrid::onFromSubset(in_data,
						from_set.device_vector(),
						to_set.device_vector(),
						full_size);
	    //OPM_THROW(std::runtime_error, "On from subset to subset is not yet implemented. We appologize for the inconvinience");
	}
    }

}  // namespace equelleCUDA



#endif // EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
