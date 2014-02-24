// This file provides unit tests for the class
// DeviceGrid, which handles grid related functionality on the GPU.

#include <boost/test/included/unit_test.hpp>
using namespace boost::unit_test;

// Include everything that the equelle runtime need
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}



void free_test_function() {
    BOOST_CHECK(2 == 1);
}

void equal_test_function(int a, int b) {
    BOOST_REQUIRE( a == b);
}


test_suite* init_unit_test_suite( int argc, char** argv )
{
    // Get user parameters
    Opm::parameter::ParameterGroup param( argc, argv, false);

    // Create the Equelle runtime
    EquelleRuntimeCUDA er(param);
    ensureRequirements(er);
    DeviceGrid dg(er.getGrid());
    std::cout << "Test: (4?) " << dg.test() << std::endl;

    framework::master_test_suite().p_name.value = "Unit test for DeviceGrid";

    int a = 1;
    int b = 1;
    //framework::master_test_suite().add( BOOST_REQUIRE( a == b ), 0);

    framework::master_test_suite().add( BOOST_TEST_CASE( &free_test_function), 1);
    
    framework::master_test_suite().add( BOOST_TEST_CASE( boost::bind(&equal_test_function, a, b)), 0);

    return 0;

}
