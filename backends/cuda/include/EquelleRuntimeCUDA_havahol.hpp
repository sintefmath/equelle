
#ifndef EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED

#include <string>
#include <fstream>
#include <iterator>
#include <thrust/host_vector.h>

#include "EquelleRuntimeCUDA.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "DeviceGrid.hpp"



// -------------------------------------------------- //
// ------------------ INPUT ------------------------- //
// -------------------------------------------------- //


template <int dummy>
equelleCUDA::CollOfScalar EquelleRuntimeCUDA::inputCollectionOfScalar( const String& name, const equelleCUDA::CollOfIndices<dummy>& coll) 
{
    std::cout << "Copy from file " << name << std::endl;
    // Copy the reading part from the CPU back-end
    const int size = coll.size();
    const bool from_file = param_.getDefault(name + "_from_file", false);
    if ( from_file ) {
	const String filename = param_.get<String>(name + "_filename");
	std::ifstream is (filename.c_str());
	if (!is) {
	    OPM_THROW(std::runtime_error, "Could not find file " << filename);
	}
	std::istream_iterator<double> begin(is);
	std::istream_iterator<double> end;
	std::vector<double> data(begin, end);
	//CollOfScalar out(data);
	//out.setValuesFromFile(begin, end);
	if ( data.size() != size) {
	    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename); 
	}
	return equelleCUDA::CollOfScalar(data);
    }
    else {
	// There is a number in the parameter file
	//CollOfScalar out(size);
	//out.setValuesUniform(param_.get<double>(name));
	//out.setValuesUniform(param_.get<double>(name), size);
	return equelleCUDA::CollOfScalar(size, param_.get<double>(name));
    }
}



template <int dummy>
equelleCUDA::CollOfIndices<dummy> EquelleRuntimeCUDA::inputDomainSubsetOf(const String& name, equelleCUDA::CollOfIndices<dummy> superset) 
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
	OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    thrust::host_vector<int> host(beg, end);
    thrust::device_vector<int> dev(host.begin(), host.end());
    //thrust::sort(dev.begin(), dev.end());
      
    equelleCUDA::CollOfIndices<dummy> subset(host);
    //subset.sort();
    
    // USING SORT ON THE DEVICE PRODUCES A STRANGE ERROR.
    // SORTING DONE ON THE HOST FOR NOW...
    // Believe this is because the _impl.hpp file is compiled by 
    // the gcc compiler as well through #includes
    
    //superset.contains(subset, name);

    // The function works for correct sets without sort and contains,
    // but it also allows for illegal input.

    return subset;
}


// ---------------------------------------------------- //
// ----------------- GRID OPERATIONS ------------------ //
// ---------------------------------------------------- //


template <int dummy>
CollOfScalar EquelleRuntimeCUDA::operatorExtend(const CollOfScalar& data_in,
						const CollOfIndices<dummy>& from_set,
						const CollOfIndices<dummy>& to_set) {
    
    if (data_in.size() != from_set.size() ) {
	OPM_THROW(std::runtime_error, "data_in (size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in Extend function."); 
    }
    if (from_set.size() > to_set.size() ) {
	OPM_THROW(std::runtime_error, "From_set (size " << from_set.size() << ") has to be a subset of to_set (size " << to_set.size() << ")");
    }
    
    return dev_grid_.operatorExtend(data_in, from_set, to_set);
}


template <int dummy>
CollOfScalar EquelleRuntimeCUDA::operatorOn(const CollOfScalar& data_in,
					    const CollOfIndices<dummy>& from_set,
					    const CollOfIndices<dummy>& to_set) {

    if ( data_in.size() != from_set.size()) {
	OPM_THROW(std::logic_error, "data_in (size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in On function for CollOfScalar.");
    }
    //if ( to_set.size() > from_set.size() ) {
    //	OPM_THROW(std::runtime_error, "To_set (size " << to_set.size() << ") has to be a subset of from_set (size " << from_set.size() << ")");
    //}

    return dev_grid_.operatorOn(data_in, from_set, to_set);
}



template<int dummy_data, int dummy_set>
CollOfIndices<dummy_data> EquelleRuntimeCUDA::operatorOn( const CollOfIndices<dummy_data>& data_in,
							  const CollOfIndices<dummy_set>& from_set,
							  const CollOfIndices<dummy_set>& to_set) {
    if ( data_in.size() != from_set.size() ) {
	OPM_THROW(std::logic_error, "data_in(size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in On function for CollOfIndices.");
    }
    return CollOfIndices<dummy_data>(dev_grid_.operatorOn(data_in, from_set, to_set) );
}



#endif // EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
