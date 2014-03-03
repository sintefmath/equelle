#define BOSOT_TEST_MAIN
#define BOOST_TEST_MODULE EquelleControllerTest

#include <memory>
#include <boost/test/unit_test.hpp>


#include <zoltan_cpp.h>
#include "EquelleRuntimeCPU.hpp"
#include "equelle/mpiutils.hpp"

BOOST_AUTO_TEST_CASE( giveGridToZoltan )
{
    Opm::parameter::ParameterGroup paramgroup;

    auto grid = equelle::createGridManager(paramgroup);

    BOOST_CHECK_EQUAL( grid->c_grid()->number_of_cells, 6 );



}

