
#ifndef EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
#define EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED



#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
//#include "DeviceGrid.hpp"
#include "wrapDeviceGrid.hpp"

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
	    int full_size;// = number_of_faces_;
	    if (dummy == 0) { // cells
	        full_size = number_of_cells_;
	    } else if (dummy == 1) {
	        full_size = number_of_faces_;
	    } else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << dummy);
	    }
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
	    int full_size; // better safe than sorry
	    if (dummy == 0) {
		full_size = number_of_cells_;
	    }
	    else if (dummy == 1) {
		full_size = number_of_faces_;
	    }
	    else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << dummy);
	    }
	    return wrapDeviceGrid::onFromSubset(in_data,
						from_set.device_vector(),
						to_set.device_vector(),
						full_size);
	    //OPM_THROW(std::runtime_error, "On from subset to subset is not yet implemented. We appologize for the inconvinience");
	}
    }



    template<int dummy_data, int dummy_set>
    thrust::device_vector<int> DeviceGrid::operatorOn( const CollOfIndices<dummy_data>& in_data,
						      const CollOfIndices<dummy_set>& from_set,
						      const CollOfIndices<dummy_set>& to_set)
    {
	std::cout << "\n\nON - COLL_OF_INDICES!!!\n\n";
	if ( from_set.isFull() ) {
	    return wrapDeviceGrid::onFromFullIndices(in_data.device_vector(),
						     to_set.device_vector());
	    
	}
	else {
	    int full_size; // better safe than sorry
	    if (dummy_set == 0) {
		full_size = number_of_cells_;
	    }
	    else if (dummy_set == 1) {
		full_size = number_of_faces_;
	    }
	    else {
		OPM_THROW(std::runtime_error, "No CollOfIndices<codim> for codim " << dummy_set);
	    }
	    return wrapDeviceGrid::onFromSubsetIndices(in_data.device_vector(),
						       from_set.device_vector(),
						       to_set.device_vector(),
						       full_size);
	    //OPM_THROW(std::runtime_error, "On from subset to subset is not yet implemented. We appologize for the inconvinience");
	}
    }
}  // namespace equelleCUDA



#endif // EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
