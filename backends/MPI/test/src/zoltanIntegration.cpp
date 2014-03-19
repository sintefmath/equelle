#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE EquelleMPIBackendTest

#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <vector>
#include <numeric>
#include <fstream>

#include <boost/test/unit_test.hpp>

#include <zoltan_cpp.h>
#include "EquelleRuntimeCPU.hpp"
#include "equelle/mpiutils.hpp"
#include "equelle/RuntimeMPI.hpp"
#include "equelle/ZoltanGrid.hpp"
#include "equelle/ZoltanGridMigrator.hpp"


using equelle::MPIInitializer;

BOOST_GLOBAL_FIXTURE( MPIInitializer );


BOOST_AUTO_TEST_CASE( gridExploration )
{
    Opm::parameter::ParameterGroup paramgroup;


    std::unique_ptr<Opm::GridManager> grid ( equelle::createGridManager(paramgroup) );

    BOOST_CHECK_EQUAL( grid->c_grid()->number_of_cells, 6 );
    equelle::dumpGrid( grid->c_grid() );

}


BOOST_AUTO_TEST_CASE( RuntimeMPI_6x1grid ) {
    equelle::RuntimeMPI runtime;

    BOOST_CHECK( runtime.zoltan != NULL );

    int ierr;
    void* grid = const_cast<void*>( reinterpret_cast<const void*>(runtime.grid_manager->c_grid() ) );


    if ( equelle::getMPIRank() == 0 ) {
        BOOST_CHECK_EQUAL( runtime.grid_manager->c_grid()->number_of_cells, 6 );
        BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfObjects( grid, &ierr ), 6 );

        // Check our querying of the 6x1 grid.
        const auto numCells = runtime.grid_manager->c_grid()->number_of_cells;
        std::vector<unsigned int> cells( numCells );
        for( unsigned int i = 0; i < numCells; ++i ) {
            cells[i] = i;
        }

        std::vector<int> numEdges( numCells, -1 );
        equelle::ZoltanGrid::getNumberOfEdgesMulti( grid, 1, 1, numCells, cells.data(), cells.data(), numEdges.data(),
                                                    &ierr );
        BOOST_CHECK_EQUAL( ierr, ZOLTAN_OK );
        BOOST_CHECK_EQUAL( numEdges[0], 1 );
        BOOST_CHECK_EQUAL( numEdges[1], 2 );
        BOOST_CHECK_EQUAL( numEdges[2], 2 );
        BOOST_CHECK_EQUAL( numEdges[3], 2 );
        BOOST_CHECK_EQUAL( numEdges[4], 2 );
        BOOST_CHECK_EQUAL( numEdges[5], 1 );


        auto totalNumberOfNeighbors = std::accumulate( numEdges.begin(), numEdges.end(), 0 );
        std::vector<ZOLTAN_ID_TYPE> edgeList( totalNumberOfNeighbors, -1 );
        std::vector<int> nbor_procs( edgeList.size() );

        equelle::ZoltanGrid::getEdgeListMulti( grid, 1, 1, numCells,
                                               cells.data(), cells.data(), numEdges.data(),
                                               edgeList.data(), nbor_procs.data(), 0, NULL, &ierr );
        // Cell 0
        BOOST_CHECK_EQUAL( edgeList[0], 1 ); // 0 is neighbor with 1

        // Cell 1
        BOOST_CHECK_EQUAL( edgeList[1], 0 );
        BOOST_CHECK_EQUAL( edgeList[2], 2 );

        // Cell 2
        BOOST_CHECK_EQUAL( edgeList[3], 1 );
        BOOST_CHECK_EQUAL( edgeList[4], 3 );

        // Cell 3
        BOOST_CHECK_EQUAL( edgeList[5], 2 );
        BOOST_CHECK_EQUAL( edgeList[6], 4 );

        // Cell 4UTO
        BOOST_CHECK_EQUAL( edgeList[7], 3 );
        BOOST_CHECK_EQUAL( edgeList[8], 5 );

        // Cell 5
        BOOST_CHECK_EQUAL( edgeList[9], 4 );

        BOOST_CHECK_EQUAL( ierr, ZOLTAN_OK );
    } else {
        BOOST_CHECK_EQUAL( runtime.grid_manager->c_grid()->number_of_cells, 0 );
        BOOST_CHECK_EQUAL( equelle::ZoltanGrid::getNumberOfObjects( grid, &ierr ), 0 );
    }

    auto zr = runtime.computePartition();
    BOOST_CHECK_EQUAL( zr.changes, 1 );

    if ( equelle::getMPIRank() == 0 ) {
        std::ofstream f("rank0-exports");
        equelle::ZoltanGrid::dumpRank0Exports( runtime.grid_manager->c_grid()->number_of_cells, zr, f );
    }
}


BOOST_AUTO_TEST_CASE( RuntimeMPI_6x2grid ) {
    equelle::RuntimeMPI runtime;
    if ( equelle::getMPIRank() == 0 ) {
        runtime.grid_manager.reset( new Opm::GridManager( 6, 2 ) );
    } // else the grid for other MPI nodes are empty in RuntimeMPI ctor.

    auto zr = runtime.computePartition();
    BOOST_CHECK_EQUAL( zr.changes, 1 );

    if ( equelle::getMPIRank() == 0 ) {
        std::ofstream f("rank0-6x2-exports");
        equelle::ZoltanGrid::dumpRank0Exports( runtime.grid_manager->c_grid()->number_of_cells, zr, f );
    }
}

BOOST_AUTO_TEST_CASE( ZoltanGridMigratorCallBackSignatures ) {
    // This test really passes when the test compiles.
    std::unique_ptr<Zoltan> zoltan( new Zoltan(MPI_COMM_WORLD) );
    using equelle::ZoltanGridMigrator;

    ZOLTAN_SAFE_CALL( zoltan->Set_Obj_Size_Fn( ZoltanGridMigrator::cellSize, NULL ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Pack_Obj_Fn( ZoltanGridMigrator::packCell, NULL ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Unpack_Obj_Fn( ZoltanGridMigrator::unpackCell, NULL ) );
}





