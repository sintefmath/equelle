#include "equelle/ZoltanGrid.hpp"

#include <stdexcept>
#include <opm/core/grid.h>

int equelle::ZoltanGrid::getNumberOfObjects(void *data, int *ierr)
{
    auto grid = reinterpret_cast<UnstructuredGrid*>( data );

    *ierr = ZOLTAN_OK;

    return grid->number_of_cells;
}

void equelle::ZoltanGrid::getCellList( void *data, int /*sizeGID*/, int /*sizeLID*/,
                                       ZOLTAN_ID_PTR globalId, ZOLTAN_ID_PTR localId,
                                       int /*wgt_dim*/, float *weights, int *ierr )
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
int equelle::ZoltanGrid::getNumberOfEdges( void *data, int num_gid_entries, int num_lid_entries,
                                           ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr )
{
    *ierr = ZOLTAN_FATAL;
    /*
    auto grid = reinterpret_cast<UnstructuredGrid*>( data );
    int cell_id = *local_id;

    int face_begin = grid->cell_facepos[cell_id];
    int face_end   = grid->cell_facepos[cell_id+1];
    int numberOfFaces = face_end - face_begin;


    */
    return 0;
}

void equelle::ZoltanGrid::getEdgeList(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, ZOLTAN_ID_PTR nbor_global_id, int *nbor_procs, int wgt_dim, float *ewgts, int *ierr)
{
    std::cerr << __FUNCTION__ << std::endl;
    *ierr = ZOLTAN_FATAL;
    //throw std::runtime_error( std::string( "Not implemented yet: " ) + __FUNCTION__ );
}
