
#include <iostream>

#include <boost/test/included/unit_test.hpp>
using namespace boost::unit_test;

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/common/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void comp_numbers(int a, int b) {
    BOOST_CHECK_EQUAL(a,b);

}

void blah() {
    std::cout << "blah()...\n";
    BOOST_CHECK_EQUAL(1,1);
}





test_suite* init_unit_test_suite( int argc, char* argv[] )
{
//int main( int argc, char** argv) {
    
    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);
    
    DeviceGrid dg(er.getGrid());
    
    framework::master_test_suite().p_name.value = "Values on grid";
    
    framework::master_test_suite().add( BOOST_TEST_CASE( &blah), 0);

    framework::master_test_suite().add( BOOST_TEST_CASE( boost::bind(comp_numbers,12,dg.number_of_cells())), 0);

    CollOfScalar = er.

    return 0;
}
