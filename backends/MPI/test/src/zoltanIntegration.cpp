#define BOSOT_TEST_MAIN
#define BOOST_TEST_MODULE EquelleControllerTest

#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>

#include <boost/test/unit_test.hpp>


#include <zoltan_cpp.h>
#include "EquelleRuntimeCPU.hpp"
#include "equelle/mpiutils.hpp"
#include "equelle/RuntimeMPI.hpp"
#include "equelle/ZoltanGrid.hpp"

void dumpGrid( const UnstructuredGrid* grid ) {
    std::stringstream centroids;
    std::stringstream face_cells;
    const auto dim = grid->dimensions;

    centroids << "Centroids: ";
    face_cells << "Face cells: ";
    for( int i = 0; i < grid->number_of_cells; ++i ) {
        centroids << "[";
        std::copy( &grid->cell_centroids[i*dim], &grid->cell_centroids[i*dim + dim],
                   std::ostream_iterator<double>( centroids, " " ) );
        centroids << "]";
    }

    for( int i = 0; i < grid->number_of_faces; ++i ) {
        face_cells << i << ": [" << grid->face_cells[2*i] << ", " << grid->face_cells[2*i + 1 ] << "], ";
    }

    std::cerr << centroids.str() << std::endl;
    std::cerr << face_cells.str();
}

BOOST_AUTO_TEST_CASE( gridExploration )
{
    Opm::parameter::ParameterGroup paramgroup;


    std::unique_ptr<Opm::GridManager> grid ( equelle::createGridManager(paramgroup) );

    BOOST_CHECK_EQUAL( grid->c_grid()->number_of_cells, 6 );
    //dumpGrid( grid->c_grid() );

}

BOOST_AUTO_TEST_CASE( RuntimeMPI_initializes_zoltan ) {
    equelle::RuntimeMPI runtime;

    BOOST_CHECK( runtime.zoltan != NULL );

    int ierr;
    void* grid = const_cast<void*>( reinterpret_cast<const void*>(runtime.grid_manager->c_grid() ) );

    if ( equelle::getMPIRank() == 0 ) {
        BOOST_CHECK_EQUAL( runtime.grid_manager->c_grid()->number_of_cells, 6 );
        BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfObjects( grid, &ierr ), 6 );

        // Check our querying of the 6x1 grid.
        unsigned int cell = 0;
        ZOLTAN_ID_PTR zptr = &cell;
        BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfEdges( grid, 1, 1, zptr, zptr, &ierr ), 1 );

        //cell = 1;
        //BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfEdges( grid, 1, 1, cell, cell, &ierr ), 2 );
        //BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfEdges( grid, 1, 1, 4, 4, &ierr ), 2 );
        //BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfEdges( grid, 1, 1, 5, 5, &ierr ), 1 );
    } else {
        BOOST_CHECK_EQUAL( runtime.grid_manager->c_grid()->number_of_cells, 0 );
        BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfObjects( grid, &ierr ), 0 );
    }

    //auto zr = runtime.computePartition();
    //BOOST_CHECK_EQUAL( zr.changes, 1 );
}





