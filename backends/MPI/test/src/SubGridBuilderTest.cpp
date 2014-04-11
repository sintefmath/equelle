#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include "opm/core/grid/GridManager.hpp"
#include "opm/core/grid.h"


#include "equelle/mpiutils.hpp"
#include "equelle/SubGridBuilder.hpp"
#include "equelle/RuntimeMPI.hpp"


BOOST_AUTO_TEST_CASE( SubGridBuilder ) {
    equelle::RuntimeMPI runtime;
    runtime.globalGrid.reset( new Opm::GridManager( 6, 1 ) );
    std::vector<int> cellsForSubGrid = { 4, 5 };

    equelle::SubGrid subGrid = equelle::SubGridBuilder::build( runtime.globalGrid->c_grid(), cellsForSubGrid );

    auto globalGrid = runtime.globalGrid->c_grid();
    auto localGrid  = subGrid.c_grid;

    //equelle::dumpGrid( runtime.grid_manager->c_grid() );

    BOOST_CHECK( !subGrid.face_global_to_local.empty() );
    BOOST_CHECK_EQUAL( subGrid.c_grid->number_of_cells, 3 );
    BOOST_CHECK_EQUAL( subGrid.number_of_ghost_cells, 1 );
    BOOST_CHECK_EQUAL( subGrid.global_face.size(), subGrid.c_grid->number_of_faces );
    BOOST_CHECK_EQUAL( subGrid.global_cell.size(), subGrid.c_grid->number_of_cells );

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

    // Check that we preserve the number of faces
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 4), equelle::GridQuerying::numFaces( localGrid, 0 ) );
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 5), equelle::GridQuerying::numFaces( localGrid, 1 ) );
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( globalGrid, 3), equelle::GridQuerying::numFaces( localGrid, 2 ) );

    // Check that the face_cell mapping is correct.
    // We now that global_face 3 is the west-side of the ghost cell (global cell 3).
    BOOST_REQUIRE_EQUAL( globalGrid->face_cells[2*3], 2 );
    BOOST_REQUIRE_EQUAL( globalGrid->face_cells[2*3 + 1], 3 );

    int newId = std::distance( subGrid.global_face.begin(), std::find( subGrid.global_face.begin(), subGrid.global_face.end(), 3 ) );
    BOOST_REQUIRE_EQUAL( localGrid->face_cells[2*newId], equelle::Boundary::inner );

    // Check that we have the right face areas for each face in the subgrid
    for( int i = 0; i <  equelle::GridQuerying::numFaces( globalGrid, 4); ++i ) {
        const int glob_startIndex = globalGrid->cell_facepos[4];
        const int loc_startIndex  = localGrid->cell_facepos[0];

        const int glob_face = globalGrid->cell_faces[glob_startIndex + i];
        const int loc_face  = localGrid->cell_faces[loc_startIndex   + i];

        BOOST_CHECK_EQUAL( globalGrid->face_areas[glob_face], localGrid->face_areas[loc_face] );

        BOOST_CHECK_EQUAL_COLLECTIONS( &(globalGrid->face_centroids[dim*glob_face]), &(globalGrid->face_centroids[dim*glob_face + dim]),
                                       &(localGrid->face_centroids[dim*loc_face]), &(localGrid->face_centroids[dim*loc_face + dim]) );

        BOOST_CHECK_EQUAL_COLLECTIONS( &(globalGrid->face_normals[dim*glob_face]), &(globalGrid->face_normals[dim*glob_face + dim]),
                                       &(localGrid->face_normals[dim*loc_face]), &(localGrid->face_normals[dim*loc_face + dim]) );

        BOOST_CHECK_EQUAL( equelle::GridQuerying::numNodes( globalGrid, glob_face ),
                           equelle::GridQuerying::numNodes( localGrid, loc_face) );

        // Check that we have copied the correct node-data
        for( int j = 0; j < equelle::GridQuerying::numNodes( globalGrid, glob_face ); ++j ) {
            int glob_node = globalGrid->face_nodes[ globalGrid->face_nodepos[glob_face] + j ];
            int loc_node  = localGrid->face_nodes[ localGrid->face_nodepos[loc_face] + j ];

            BOOST_CHECK_EQUAL_COLLECTIONS( &(globalGrid->node_coordinates[dim*glob_node]), &(globalGrid->node_coordinates[dim*glob_node + dim]),
                                           &(localGrid->node_coordinates[dim*loc_node]),   &(localGrid->node_coordinates[dim*loc_node + dim]) );
        }

    }

    //destroy_grid( subGrid.c_grid );
}

BOOST_AUTO_TEST_CASE( SubGridArrays ) {
    equelle::RuntimeMPI runtime;
    runtime.globalGrid.reset( new Opm::GridManager( 6, 1 ) );
    std::vector<int> cellsForSubGrid = { 4, 5 };

    equelle::SubGrid subGrid = equelle::SubGridBuilder::build( runtime.globalGrid->c_grid(), cellsForSubGrid );

    BOOST_CHECK_EQUAL( subGrid.global_cell.size(), subGrid.cell_global_to_local.size() );
    BOOST_CHECK_EQUAL( subGrid.global_face.size(), subGrid.face_global_to_local.size()  );

    for( auto x: subGrid.cell_global_to_local )  {
        BOOST_CHECK_EQUAL( x.first, subGrid.global_cell[x.second] );
    }

    for( auto x: subGrid.face_global_to_local ) {
        BOOST_CHECK_EQUAL( x.first, subGrid.global_face[x.second] );
    }

}

BOOST_AUTO_TEST_CASE( GridQueryingFunctions ) {
    equelle::RuntimeMPI runtime;

    // Our well known 6x1 grid
    runtime.globalGrid.reset( new Opm::GridManager( 6, 1 ) );

    auto grid = runtime.globalGrid->c_grid();
    BOOST_CHECK_EQUAL( equelle::GridQuerying::numFaces( grid, 0), 4 );
}
