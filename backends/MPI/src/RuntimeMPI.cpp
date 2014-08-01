#include "equelle/RuntimeMPI.hpp"
#include <iostream>
#include <fstream>

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

std::string logfilename() {
    std::stringstream ss;
    ss << "runtimempi-" << equelle::getMPIRank() << ".log";
    return ss.str();
}

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
    : logstream( logfilename() )
{     
    param_.disableOutput();
    initializeZoltan();
    initializeGrid();
}

RuntimeMPI::RuntimeMPI(const Opm::parameter::ParameterGroup &param)
    : logstream( logfilename() ),
      param_( param )

{
    param_.disableOutput();
    initializeZoltan();
    globalGrid.reset( equelle::createGridManager( param_ ) );

    logstream << "Hello from rank " << equelle::getMPIRank() << std::endl;
}

RuntimeMPI::~RuntimeMPI()
{
    // Zoltan resources must be deleted before we call MPI_Finalize.
    zoltan.release();
}

void RuntimeMPI::decompose()
{
    auto startTime = MPI_Wtime();

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

    auto endTime = MPI_Wtime();

    logstream << "Decomposing took " << endTime-startTime << " seconds\n";
    logstream << "subGrid.number_of_ghost_cells: " << subGrid.number_of_ghost_cells << std::endl;
    logstream << "subGrid.global_cell.size(): " << subGrid.cell_local_to_global.size() << std::endl;
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

CollOfFace RuntimeMPI::allFaces() const
{
    return runtime->allFaces();
}

CollOfCell RuntimeMPI::boundaryCells() const
{
    CollOfCell cells;
    return runtime->boundaryCells();
}

CollOfFace RuntimeMPI::boundaryFaces() const
{
    CollOfFace boundary;

    for( int i = 0; i < subGrid.c_grid->number_of_faces; ++i ) {
        if (subGrid.c_grid->face_cells[2*i] == Boundary::outer || subGrid.c_grid->face_cells[(2*i)+1] == Boundary::outer ) {
            boundary.emplace_back( i );
        }
    }

    return boundary;
}

CollOfScalar RuntimeMPI::inputCollectionOfScalar(const String& /* name */, const CollOfFace & /* coll */ )
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
            auto glob = subGrid.cell_local_to_global[i];

            localData[i] = data[glob];
        }

        return CollOfScalar(CollOfScalar::V(Eigen::Map<CollOfScalar::V>(&localData[0], size)));
    } else {
        // Uniform values.
        return CollOfScalar(CollOfScalar::V::Constant(size, param_.get<double>(name)));
    }

}

CollOfFace RuntimeMPI::inputDomainSubsetOf(const String &name, const CollOfFace &superset)
{
    // This implementation is based on a copy of EquelleRuntimeCPU::inputDomainSubsetOf
    // but we rewrite the indices into our local index-space.
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;

    CollOfFace data;
    for (auto it = beg; it != end; ++it) {
        logstream << "Read " << *it << std::endl;
        auto jt = subGrid.face_global_to_local.find( *it );
        if ( jt != subGrid.face_global_to_local.end() ) { // This face is part of our domain
            data.emplace_back( jt->second );

            logstream << "Adding " << *it << " -> " << jt->second << std::endl;
        } // else the face is not part of our domain
    }

    // Needed to allow for std::includes to give valid results.
    std::sort( data.begin(), data.end() );

    if (!includes(superset.begin(), superset.end(), data.begin(), data.end())) {
        logstream << "Rank: " << equelle::getMPIRank() << " is throwing." << std::endl;
        OPM_THROW(std::runtime_error, "Given faces are not in the assumed subset.");
    }

    return data;
}

CollOfCell RuntimeMPI::inputDomainSubsetOf(const String &name, const CollOfCell &superset)
{
    // This implementation is based on a copy of EquelleRuntimeCPU::inputDomainSubsetOf
    // but we rewrite the indices into our local index-space.
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;

    CollOfCell data;
    for (auto it = beg; it != end; ++it) {
        logstream << "Read " << *it << std::endl;
        auto jt = subGrid.cell_global_to_local.find( *it );
        if ( jt != subGrid.cell_global_to_local.end() ) { // This cell is part of our domain
            data.emplace_back( jt->second );

            logstream << "Adding " << *it << " -> " << jt->second << std::endl;
        } // else the cell is not part of our domain
    }

    // Needed to allow for std::includes to give valid results.
    std::sort( data.begin(), data.end() );

    if (!includes(superset.begin(), superset.end(), data.begin(), data.end())) {
        logstream << "Rank: " << equelle::getMPIRank() << " is throwing." << std::endl;
        OPM_THROW(std::runtime_error, "Given cells are not in the assumed subset.");
    }

    return data;
}


Scalar RuntimeMPI::inputScalarWithDefault(const String &name, const Scalar default_value)
{
    return runtime->inputScalarWithDefault( name, default_value );
}

void RuntimeMPI::output(const String &tag, const CollOfScalar &vals)
{
    auto val = allGather( vals );
    if ( equelle::getMPIRank() == 0 ) {
        runtime->output( tag, val );
    }
}

equelle::CollOfScalar equelle::RuntimeMPI::allGather( const equelle::CollOfScalar &coll )
{
    // Get the size of the collection on every node
    struct CollOfScalarSize {
        int size = 0;
        int numBlocks = 0;
    };

    int rank = equelle::getMPIRank();
    int world_size = equelle::getMPISize();
    std::vector<CollOfScalarSize> sizes( world_size );

    sizes[rank].size = coll.size();
    sizes[rank].numBlocks = coll.numBlocks();

    MPI_SAFE_CALL( MPI_Allgather( &sizes[rank], 2, MPI_INT,
                                  sizes.data(), 2, MPI_INT, MPI_COMM_WORLD ) );

    // Get the globalgids for the elements in coll.
    // NB. These globalgids are the global ID for the collection! As the serial backend would see them.

    std::vector<int> recvcounts( world_size); // Number of elements in each node
    std::vector<int> displacements( world_size +1 ); // Displacements where each node store the data

    for( int i = 0; i < equelle::getMPISize(); ++i ) {
        recvcounts[i] = sizes[i].size;
        displacements[i+1] = displacements[i] + sizes[i].size;
    }

    // Total number of elements, including duplicates
    const int total_number_of_elements = displacements.back();
    std::vector<int> global_id_mapping( total_number_of_elements );

    // We do not need to copy the local data into the global structure.
    // That is handeled by MPI_Allgatherv
    MPI_SAFE_CALL(
        MPI_Allgatherv( subGrid.cell_local_to_global.data(), subGrid.cell_local_to_global.size(), MPI_INT,
                        global_id_mapping.data(), recvcounts.data(), displacements.data(),
                        MPI_INT, MPI_COMM_WORLD ) );

    // Send the value part of an AutoDiffBlock, collect them in a global array
    // offsets are the same as for the global_id_mapping
    std::vector<double> adbvalues( total_number_of_elements );

    MPI_SAFE_CALL( MPI_Allgatherv( const_cast<double*>( coll.value().data() ), coll.value().size(), MPI_DOUBLE,
                     adbvalues.data(), recvcounts.data(), displacements.data(),
                     MPI_DOUBLE, MPI_COMM_WORLD ) );

    int unique_number_of_elements = 1 + *std::max_element( global_id_mapping.begin(), global_id_mapping.end() );

    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> V;
    V v_new( unique_number_of_elements );

    for( int i = 0; i < total_number_of_elements; ++i ) {
        int global_id = global_id_mapping[i];
        v_new[global_id] = adbvalues[i];
    }

    return CollOfScalar( v_new );
}

} // namespace equlle
