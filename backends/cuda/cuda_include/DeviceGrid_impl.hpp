
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
	
	std::cout << "\nEXTEND\n\n";
	

	return CollOfScalar(to_set.size(),0);
    }


}


#endif // EQUELLE_DEVICEGRID_IMPL_HEADER_INCLUDED
