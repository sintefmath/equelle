#include "equelle/ZoltanGrid.hpp"

#include <algorithm>
#include <stdexcept>
#include <iterator>
#include <ostream>
#include <fstream>

#include <opm/core/grid.h>
#include <opm/autodiff/AutoDiffHelpers.hpp>

#include "equelle/mpiutils.hpp"

int equelle::ZoltanGrid::getNumberOfObjects(void *data, int *ierr)
{
    auto grid = reinterpret_cast<UnstructuredGrid*>( data );

    *ierr = ZOLTAN_OK;

    return grid->number_of_cells;
}

void equelle::ZoltanGrid::getCellList( void *data, int /*sizeGID*/, int /*sizeLID*/,
                                       ZOLTAN_ID_PTR globalId, ZOLTAN_ID_PTR localId,
                                       int /*wgt_dim*/, float* /* weights */, int *ierr )
{
    *ierr = ZOLTAN_OK;
    auto grid = reinterpret_cast<UnstructuredGrid*>( data );

    for( auto i = 0; i < grid->number_of_cells; ++i ) {
        globalId[i] = i;
        localId[i]  = i;
        //weights[i]  = 1.0f;
    }
}

// Should we call it getNumberOfNeighbors (for a given cell?) In other words, the number of interior edges.
void equelle::ZoltanGrid::getNumberOfEdgesMulti( void *data, int /* num_gid_entries */, int /* num_lid_entries */,  int num_obj,
                                           ZOLTAN_ID_PTR  /*global_id*/  , ZOLTAN_ID_PTR /* local_id */ , int* numEdges, int *ierr )
{
    *ierr = ZOLTAN_FATAL;

    auto grid = reinterpret_cast<UnstructuredGrid*>( data );


    // The connectivity graph contains the number of neighbors for each cell on the diagonal
    // and the index of the nonzero, and equal to -1, entries of the row contains the neighboring cell ID.
    // Proof is by assuming c_1 is neighbor to c_2.
    // - On the diagnoal this sums the number of neighbors.
    // - On the nonzero entries this is always -1 due to the orientation of the edges (always 1*(-1) or vice versa).
    Opm::HelperOps helperOps( *grid );
    Opm::HelperOps::M adj = helperOps.div * helperOps.ngrad;

    auto diag = adj.diagonal();
    for( int i = 0; i < num_obj; ++i ) {
        numEdges[i] = diag[i];
    }

    *ierr = ZOLTAN_OK;
}

void equelle::ZoltanGrid::getEdgeListMulti(void *data, int /* num_gid_entries */, int /* num_lid_entries */, int num_obj,
                                           ZOLTAN_ID_PTR /* global_ids */, ZOLTAN_ID_PTR /* local_ids */ , int *num_edges,
                                           ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs,
                                           int /* wgt_dim */ , float */* ewgts */, int *ierr )
{
    *ierr = ZOLTAN_FATAL;
    auto grid = reinterpret_cast<UnstructuredGrid*>( data );
    assert( num_obj == grid->number_of_cells );

    // The connectivity graph contains the number of neighbors for each cell on the diagonal
    // and the index of the nonzero, and equal to -1, entries of the row contains the neighboring cell ID.
    // Proof is by assuming c_1 is neighbor to c_2.
    // - On the diagnoal this sums the number of neighbors.
    // - On the nonzero entries this is always -1 due to the orientation of the edges (always 1*(-1) or vice versa).
    Opm::HelperOps helperOps( *grid );
    Opm::HelperOps::M adj = helperOps.div * helperOps.ngrad;

    // The global id of the neighbors of a cell are given by the column-index of the nonzero, non-diagonal, entries of each row.
    // These values are always -1.
    int global_offset = 0;
    for( int k = 0; k < adj.outerSize(); ++k ) {
        for( Eigen::SparseMatrix<double>::InnerIterator it( adj, k); it; ++it ) {                       
            if( it.row() == it.col() ) { // Skip diagonal elements
                assert( it.value() == num_edges[k] );
            } else {
                // By using it.index() we can ignore what is the inner and outer dimensions. (Ie col-major or row-major.)
                nbor_global_id[global_offset] = it.index();
                nbor_procs[global_offset] = equelle::getMPIRank();
                global_offset++;
            }
        }       
    }    

    *ierr = ZOLTAN_OK;
}

void equelle::ZoltanGrid::dumpRank0Exports( int numCells, const equelle::zoltanReturns& zr, std::ostream& out)
{
    std::vector<int> v( numCells, 0 ); // By default all nodes belong to rank 0.

    for( int i = 0; i < zr.numExport; ++i ) {
        v[ zr.exportGlobalGids[i] ] = zr.exportProcs[i];
    }
    std::copy( v.begin(), v.end(), std::ostream_iterator<int>( out, " " ) );
    std::cout << std::endl;


    /* // Might still be useful just to see where they end up.
    std::copy_n( zr.exportGlobalGids, zr.numExport, std::ostream_iterator<int>( out, " ") );
    out << std::endl;
    std::copy_n( zr.exportProcs, zr.numExport,  std::ostream_iterator<int>( out, " ") );
    out << std::endl;
    */
}
