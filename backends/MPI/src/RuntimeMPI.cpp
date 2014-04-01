#include "equelle/RuntimeMPI.hpp"
#include <iostream>

#include <mpi.h>

#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <zoltan_cpp.h>
#pragma GCC diagnostic pop

#include <opm/core/grid/GridManager.hpp>
#include <boost/iterator/counting_iterator.hpp>

#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/mpiutils.hpp"
#include "equelle/SubGridBuilder.hpp"


namespace equelle {

void RuntimeMPI::initializeZoltan()
{
    zoltan.reset( new Zoltan( MPI_COMM_WORLD ) );

    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "DEBUG_LEVEL", "0" ) );

#ifdef EQUELLE_DEBUG
    // Check that the query functions return valid input data; 0 or 1. (This slows performance; intended for debugging.)
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "CHECK_HYPERGRAPH", "1" ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "DEBUG_LEVEL", "2" ) );
#endif

    // Use hypergraph partitioning
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "LB_METHOD", "GRAPH" ) );
    // Partition everything without concern for cost.
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "LB_APPROACH", "PARTITION" ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Param( "PHG_EDGE_SIZE_THRESHOLD", "1.0" ) );
}

void RuntimeMPI::initializeGrid()
{
    globalGrid.reset( new Opm::GridManager( 6, 1 ) );

}

RuntimeMPI::RuntimeMPI()
{     
    param_.disableOutput();
    initializeZoltan();
    initializeGrid();
}

RuntimeMPI::RuntimeMPI(const Opm::parameter::ParameterGroup &param)
    : param_( param )
{
    param_.disableOutput();
    initializeZoltan();
    globalGrid.reset( equelle::createGridManager( param_ ) );
}

RuntimeMPI::~RuntimeMPI()
{
    // Zoltan resources must be deleted before we call MPI_Finalize.
    zoltan.release();
}

void RuntimeMPI::decompose()
{
    auto zr = computePartition();
    std::vector<int> localCells;

    if ( getMPIRank() == 0 ) {
        // Node 0 must compute which cells not to export.
        std::set_difference( boost::counting_iterator<int>(0), boost::counting_iterator<int>( globalGrid->c_grid()->number_of_cells ),
                             zr.exportGlobalGids, zr.exportGlobalGids + zr.numExport, std::back_inserter( localCells ) );
    } else {
        localCells.resize( zr.numImport );
        std::copy_n( zr.importGlobalGids, zr.numImport, localCells.begin() );
    }

    subGrid = SubGridBuilder::build( globalGrid->c_grid(), localCells );

    runtime.reset( new EquelleRuntimeCPU( subGrid.c_grid, param_ ) );
}

zoltanReturns RuntimeMPI::computePartition()
{
    zoltanReturns zr;
    void* grid;
    Opm::GridManager emptyGrid( 0, 0 );
    // Let non rank-0 nodes pass in the empty grid here.l

    if ( getMPIRank() == 0 ) {
        grid = const_cast<void*>( reinterpret_cast<const void*>( globalGrid->c_grid() ) );
    } else {
        grid = const_cast<void*>( reinterpret_cast<const void*>( emptyGrid.c_grid()) );
    }

    ZOLTAN_SAFE_CALL( zoltan->Set_Num_Obj_Fn( ZoltanGrid::getNumberOfObjects, grid ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Obj_List_Fn( ZoltanGrid::getCellList, grid ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Num_Edges_Multi_Fn( ZoltanGrid::getNumberOfEdgesMulti, grid ) );
    ZOLTAN_SAFE_CALL( zoltan->Set_Edge_List_Multi_Fn( ZoltanGrid::getEdgeListMulti, grid ) );

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

CollOfCell RuntimeMPI::allCells() const
{
    return runtime->allCells();
}

CollOfScalar RuntimeMPI::inputCollectionOfScalar(const String &name, const CollOfFace &coll)
{
    throw std::runtime_error("Not implemented");
}

CollOfScalar RuntimeMPI::inputCollectionOfScalar(const String &name, const CollOfCell &coll)
{
    const int size = coll.size();
    const bool from_file = param_.getDefault(name + "_from_file", false);
    if (from_file) {
        const String filename = param_.get<String>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            OPM_THROW(std::runtime_error, "Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;
        std::vector<double> data(beg, end);

        std::vector<double> localData( coll.size() );

        // Map into local cell enumeration
        for( int i = 0; i < coll.size(); ++i ) {
            auto glob = subGrid.global_cell[i];

            localData[i] = data[glob];
        }

        return CollOfScalar(CollOfScalar::V(Eigen::Map<CollOfScalar::V>(&localData[0], size)));
    } else {
        // Uniform values.
        return CollOfScalar(CollOfScalar::V::Constant(size, param_.get<double>(name)));
    }

}


} // namespace equlle
