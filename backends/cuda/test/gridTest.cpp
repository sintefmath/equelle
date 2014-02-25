// This file provides unit tests for the class
// DeviceGrid, which handles grid related functionality on the GPU.

//#include <boost/test/included/unit_test.hpp>
//using namespace boost::unit_test;

// CUDA code and BOOST don't mix. Need to write tests without the BOOST framework.

#include <cuda.h>
#include <cuda_runtime.h>

#include "gridTest.h"

// Include everything that the equelle runtime need
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>

#include <opm/core/utility/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}

//int num_errors;

void free_test_function() {
    
}

void equal_test_function(int a, int b) {
    if ( a != b ) {
	OPM_THROW(std::runtime_error, "\nequal_test_function - " << a << " != " << b);
    }
}


int main( int argc, char** argv )
{
    // Get user parameters
    Opm::parameter::ParameterGroup param( argc, argv, false);

    // Create the Equelle runtime
    EquelleRuntimeCUDA er(param);
    ensureRequirements(er);
    
    // Get the device grid so that we can play around with it!
    DeviceGrid dg(er.getGrid());
       
    int a = 1;
    int b = 1;
    //framework::master_test_suite().add( BOOST_REQUIRE( a == b ), 0);

    free_test_function();
    
    equal_test_function(a, b);


    //int out = cuda_main(dg);
    //std::cout << "Back in main!\n";
    //return out;
    return cuda_main(dg);
}
