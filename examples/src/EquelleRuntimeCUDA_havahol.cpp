#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_cuda.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"

#include <string>
#include <iostream>

//
//   CPP for implementation of non-cuda functions
//


void EquelleRuntimeCUDA::output(const String& tag, const CollOfScalar& coll, int dummy)
{
    // Get data back to host
    coll.copyToHost();
    
    // Do this with the device_vector instead!
    
    std::cout << "\n";
    std::cout << "Values in " << tag << std::endl;
    for(int i = 0; i < coll.getSize(); ++i) {
	std::cout << coll.getValue(i) << "  ";
    }
    std::cout << std::endl;
}
