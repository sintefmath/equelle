#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_cuda.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"

#include <string>
#include <iostream>
#include <thrust/host_vector.h>

//
//   CPP for implementation of non-cuda functions
//


void EquelleRuntimeCUDA::output(const String& tag, CollOfScalar coll, int dummy)
{
    // Get data back to host
    double* host_vals = coll.getHostValues();
    
    // Do this with the device_vector instead!
    
    std::cout << "\n";
    std::cout << "The " << coll.getSize() << " values in " << tag << std::endl;
    for(int i = 0; i < coll.getSize(); ++i) {
	std::cout << host_vals[i] << "  ";
    }
    std::cout << std::endl;
}
