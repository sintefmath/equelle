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

    auto globalGrid = runtime.grid_manager->c_grid();
    auto localGrid  = subGrid.c_grid;

    equelle::dumpGrid( runtime.grid_manager->c_grid() );

    BOOST_CHECK_EQUAL( subGrid.c_grid->number_of_cells, 3 );
    BOOST_CHECK_EQUAL( subGrid.number_of_ghost_cells, 1 );

    // Check the local to global mapping
    BOOST_CHECK_EQUAL( 4, subGrid.global_cell[0] );
    BOOST_CHECK_EQUAL( 5, subGrid.global_cell[1] );
    BOOST_CHECK_EQUAL( 3, subGrid.global_cell[2] );

    const int dim = subGrid.c_grid->dimensions;

    // Check copying of centroids
    BOOST_CHECK_EQUAL( localGrid->cell_centroids[(dim*0)+0], globalGrid->cell_centroids[(4*dim)+0] );
    BOOST_CHECK_EQUAL( localGrid->cell_centroids[(dim*0)+1], globalGrid->cell_centroids[(4*dim)+1] );

    BOOST_CHECK_EQUAL( localGrid->cell_centroids[(dim*1)+0], globalGrid->cell_centroids[(5*dim)+0] );
    BOOST_CHECK_EQUAL( localGrid->cell_centroids[(dim*1)+1], globalGrid->cell_centroids[(5*dim)+1] );

    BOOST_CHECK_EQUAL( subGrid.c_grid->cell_centroids[(dim*2)+0], globalGrid->cell_centroids[(3*dim)+0] );
    BOOST_CHECK_EQUAL( subGrid.c_grid->cell_centroids[(dim*2)+1], globalGrid->cell_centroids[(3*dim)+1] );

    // Check copying of the cell volumes
    BOOST_CHECK_EQUAL( localGrid->cell_volumes[0], globalGrid->cell_volumes[4] );
    BOOST_CHECK_EQUAL( localGrid->cell_volumes[1], globalGrid->cell_volumes[5] );
    BOOST_CHECK_EQUAL( localGrid->cell_volumes[2], globalGrid->cell_volumes[3] );
/*
    // Check that we preserve the number of faces
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 4), equelle::GridQuerying::numFaces( localGrid, 0 ) );
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 5), equelle::GridQuerying::numFaces( localGrid, 1 ) );
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 3), equelle::GridQuerying::numFaces( localGrid, 2 ) );
*/
    destroy_grid( subGrid.c_grid );
}

BOOST_AUTO_TEST_CASE( GridQueryingFunctions ) {
    equelle::RuntimeMPI runtime;

    // Our well known 6x1 grid
    runtime.grid_manager.reset( new Opm::GridManager( 6, 1 ) );

    auto grid = runtime.grid_manager->c_grid();
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( grid, 0), 4 );
}
