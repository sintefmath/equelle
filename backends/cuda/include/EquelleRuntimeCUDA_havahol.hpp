
#ifndef EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED

#include <string>
#include <fstream>
#include <iterator>

#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_cuda.hpp"

template <class SomeCollection>
CollOfScalar EquelleRuntimeCUDA::inputCollectionOfScalar( const String& name, const SomeCollection& coll) 
{
    std::cout << "Copy from file " << name << std::endl;
    // Copy the reading part from the CPU back-end
    const int size = coll.size();
    const bool from_file = param_.getDefault(name + "_from_file", false);
    //if ( from_file) {
    if ( from_file ) {
	const String filename = param_.get<String>(name + "_filename");
	std::ifstream is (filename.c_str());
	if (!is) {
	    OPM_THROW(std::runtime_error, "Could not find file " << filename);
	}
	std::istream_iterator<double> begin(is);
	std::istream_iterator<double> end;
	//CollOfScalar *out = new CollOfScalar(size);
	CollOfScalar out(size);
	out.setValuesFromFile(begin, end);
	if ( out.size() != size) {
	    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename); 
	}
	return out;
    }
    else {
	// There is a number in the parameter file
	CollOfScalar out(size);
	out.setValuesUniform(param_.get<double>(name));
	//out.setValuesUniform(param_.get<double>(name), size);
	return out;
    }
}



#endif // EQUELLERUNTIMECUDA_HAVAHOL_HEADER_INCLUDED
