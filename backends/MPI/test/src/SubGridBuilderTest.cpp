#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include "opm/core/grid/GridManager.hpp"
#include "opm/core/grid.h"


#include "equelle/mpiutils.hpp"
#include "equelle/SubGridBuilder.hpp"
#include "equelle/RuntimeMPI.hpp"


BOOST_AUTO_TEST_CASE( SubGridBuilder ) {
    equelle::RuntimeMPI runtime;
    runtime.grid_manager.reset( new Opm::GridManager( 6, 1 ) );
    std::vector<int> cellsForSubGrid = { 4, 5 };

    equelle::SubGrid subGrid = equelle::SubGridBuilder::build( runtime.grid_manager->c_grid(), cellsForSubGrid );

    //equelle::dumpGrid( runtime.grid_manager->c_grid() );

    BOOST_CHECK_EQUAL( subGrid.number_of_ghost_cells, 1 );

    // Check the local to global mapping
    BOOST_CHECK_EQUAL( 4, subGrid.global_cell[0] );
    BOOST_CHECK_EQUAL( 5, subGrid.global_cell[1] );

}
