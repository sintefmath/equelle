#include "equelle/RuntimeMPI.hpp"
#include <mutex>
#include <iostream>

#include <mpi.h>
#include <zoltan_cpp.h>
#include <opm/core/grid/GridManager.hpp>

#include "EquelleRuntimeCPU.hpp"
#include "equelle/mpiutils.hpp"


namespace equelle {

namespace impl {
void initMPIandZoltan() {
    MPI_SAFE_CALL( MPI_Init( NULL, NULL ) );

    int size;
    MPI_SAFE_CALL( MPI_Comm_size( MPI_COMM_WORLD, &size ) );

    float zoltanVersion;
    ZOLTAN_SAFE_CALL( Zoltan_Initialize( 0, NULL, &zoltanVersion ) );
}
}

std::once_flag flag;

void RuntimeMPI::initializeZoltan()
{
    zoltan.reset( new Zoltan( MPI_COMM_WORLD ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "DEBUG_LEVEL", "2" ) );
    // Use hypergraph partitioning
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "LB_METHOD", "GRAPH" ) );
    // Partition everything without concern for cost.
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "LB_APPROACH", "PARTITION" ) );
}

void RuntimeMPI::initializeGrid()
{
    if ( getMPIRank() == 0 ) {
        grid_manager.reset( new Opm::GridManager( 6, 1 ) );
    } else {
        grid_manager.reset( new Opm::GridManager( 0, 0 ) );
    }
}

RuntimeMPI::RuntimeMPI()
{
    std::call_once( flag, impl::initMPIandZoltan );
    initializeZoltan();
    initializeGrid();
}

RuntimeMPI::~RuntimeMPI()
{
    // Zoltan resources must be deleted before we call MPI_Finalize.
    zoltan.release();
    MPI_SAFE_CALL( MPI_Finalize() );

}

zoltanReturns RuntimeMPI::computePartition()
{
    zoltanReturns zr;

    void* grid = const_cast<void*>( reinterpret_cast<const void*>( grid_manager->c_grid() ) );

    zoltan->Set_Num_Obj_Fn( ZoltanGrid::getNumberOfObjects, grid );
    zoltan->Set_Obj_List_Fn( ZoltanGrid::getCellList, grid );
    zoltan->Set_Num_Edges_Fn( ZoltanGrid::getNumberOfEdges, grid );
    zoltan->Set_Edge_List_Fn( ZoltanGrid::getEdgeList, grid );

    ZOLTAN_SAFE_CALL(
                       zoltan->LB_Partition( zr.changes,         /* 1 if partitioning was changed, 0 otherwise */
                                            zr.numGidEntries,   /* Number of integers used for a global ID */
                                            zr.numLidEntries,   /* Number of integers used for a local ID */
                                            zr.numImport,       /* Number of vertices to be sent to me */
                                            zr.importGlobalGids,/* Global IDs of vertices to be sent to me */
                                            zr.importLocalGids, /* Local IDs of vertices to be sent to me */
                                            zr.importProcs,     /* Process rank for source of each incoming vertex */
                                            zr.importToPart,    /* New partition for each incoming vertex */
                                            zr.numExport,       /* Number of vertices I must send to other processes*/
                                            zr.exportGlobalGids,/* Global IDs of the vertices I must send */
                                            zr.exportLocalGids, /* Local IDs of the vertices I must send */
                                            zr.exportProcs,     /* Process to which I send each of the vertices */
                                            zr.exportToPart ) );  /* Partition to which each vertex will belong */

    return zr;
}


} // namespace equlle
