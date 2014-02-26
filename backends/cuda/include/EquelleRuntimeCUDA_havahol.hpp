
#ifndef EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED

#include <string>
#include <fstream>
#include <iterator>
#include <vector>

#include "EquelleRuntimeCUDA.hpp"
#include "CollOfScalar.hpp"




template <class SomeCollection>
equelleCUDA::CollOfScalar EquelleRuntimeCUDA::inputCollectionOfScalar( const String& name, const SomeCollection& coll) 
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


#endif // EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
