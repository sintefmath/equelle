#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE SubGridValidator

#include <cstdlib>
#include <iostream>
#include <vector>
#include <algorithm>

#include <boost/iterator/counting_iterator.hpp>

#include <opm/core/grid/GridManager.hpp>
#include <opm/core/grid.h>

#include "equelle/mpiutils.hpp"
#include "equelle/RuntimeMPI.hpp"
#include "equelle/SubGridBuilder.hpp"

#include <boost/test/unit_test.hpp>

using equelle::MPIInitializer;

BOOST_GLOBAL_FIXTURE( MPIInitializer );

BOOST_AUTO_TEST_CASE( checkSubGrids ) {
    using std::begin;
    using std::end;

    std::string filename("norne.grid");

    equelle::RuntimeMPI runtime;
    runtime.globalGrid.reset( new Opm::GridManager( filename ) );

    auto zr = runtime.computePartition();

    if ( equelle::getMPIRank() == 0 ) {
        const int numPartitions = equelle::getMPISize();
        const int numberOfCells = runtime.globalGrid->c_grid()->number_of_cells;
        std::vector<std::vector<int>> exports( numPartitions );

        for( int i = 0; i < zr.numExport; ++i ) {
            exports[ zr.exportProcs[i] ].push_back( zr.exportGlobalGids[i] );
        }

        BOOST_ASSERT( std::is_sorted( zr.exportGlobalGids, zr.exportGlobalGids + zr.numExport ) );

        // Build the global cell ids for the rank 0 node.
        std::set_difference( boost::counting_iterator<int>(0), boost::counting_iterator<int>( numberOfCells ),
                             zr.exportGlobalGids, zr.exportGlobalGids + zr.numExport, std::back_inserter( exports[0] ) );

        for( int i = 1; i < exports.size(); ++i ) {
            std::vector<int> v;
            std::set_intersection( exports[0].begin(), exports[0].end(),
                    exports[i].begin(), exports[i].end(),
                    std::back_inserter( v ) );
            BOOST_ASSERT( v.empty() );
        }

        BOOST_MESSAGE( "Sizes of each partition:" );
        for( int i = 0; i < exports.size(); ++i ) {
            BOOST_MESSAGE( i << ".size() == " << exports[i].size() );
        }

        for( auto& v: exports ) {
            auto subGrid = equelle::SubGridBuilder::build( runtime.globalGrid->c_grid(), v );

            BOOST_CHECK_EQUAL( subGrid.c_grid->number_of_cells - subGrid.number_of_ghost_cells, v.size() );

            auto globalGrid = runtime.globalGrid->c_grid();
            auto localGrid = subGrid.c_grid;
            auto dim = globalGrid->dimensions;

            for( int cell = 0; cell < subGrid.c_grid->number_of_cells; ++cell ) {
                const int lid = cell;
                const int gid = subGrid.cell_local_to_global[cell];

                BOOST_CHECK_EQUAL_COLLECTIONS(
                            &(globalGrid->cell_centroids[dim*gid]), &(globalGrid->cell_centroids[dim*gid + dim]),
                            &(localGrid->cell_centroids[dim*lid]), &(localGrid->cell_centroids[dim*lid + dim]) );

                BOOST_CHECK_EQUAL( globalGrid->cell_volumes[gid], localGrid->cell_volumes[lid]);

                BOOST_REQUIRE_EQUAL( equelle::GridQuerying::numFaces( globalGrid, gid ),
                                   equelle::GridQuerying::numFaces( localGrid, lid ) );

                for( int i = 0; i < equelle::GridQuerying::numFaces( globalGrid, gid ); ++i ) {
                    const int glob_startIndex = globalGrid->cell_facepos[gid];
                    const int loc_startIndex  = localGrid->cell_facepos[lid];

                    const int glob_face = globalGrid->cell_faces[glob_startIndex + i];
                    const int loc_face  = localGrid->cell_faces[loc_startIndex   + i];

                    BOOST_REQUIRE_EQUAL( globalGrid->face_areas[glob_face], localGrid->face_areas[loc_face] );

                    BOOST_REQUIRE_EQUAL_COLLECTIONS( &(globalGrid->face_centroids[dim*glob_face]), &(globalGrid->face_centroids[dim*glob_face + dim]),
                                                     &(localGrid->face_centroids[dim*loc_face]), &(localGrid->face_centroids[dim*loc_face + dim]) );

                    BOOST_REQUIRE_EQUAL_COLLECTIONS( &(globalGrid->face_normals[dim*glob_face]), &(globalGrid->face_normals[dim*glob_face + dim]),
                                                     &(localGrid->face_normals[dim*loc_face]), &(localGrid->face_normals[dim*loc_face + dim]) );

                    BOOST_REQUIRE_EQUAL( equelle::GridQuerying::numNodes( globalGrid, glob_face ),
                                         equelle::GridQuerying::numNodes( localGrid, loc_face) );

                    // Check that we have copied the correct node-data
                    for( int j = 0; j < equelle::GridQuerying::numNodes( globalGrid, glob_face ); ++j ) {
                        int glob_node = globalGrid->face_nodes[ globalGrid->face_nodepos[glob_face] + j ];
                        int loc_node  = localGrid->face_nodes[ localGrid->face_nodepos[loc_face] + j ];

                        BOOST_REQUIRE_EQUAL_COLLECTIONS( &(globalGrid->node_coordinates[dim*glob_node]), &(globalGrid->node_coordinates[dim*glob_node + dim]),
                                                         &(localGrid->node_coordinates[dim*loc_node]),   &(localGrid->node_coordinates[dim*loc_node + dim]) );
                    }
                }

            }

        }

    }

}
