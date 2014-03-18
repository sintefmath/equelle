#pragma once

#include <zoltan_cpp.h>

namespace equelle {

/**
 *  zoltanReturns holds all variables that are returen by pointer/reference from Zoltan::LB_Partition.
 *  This is merely a convenience struct that is handy to pass around.
 */
struct zoltanReturns {
    int changes{}, numGidEntries{}, numLidEntries{}, numImport{}, numExport{};
    ZOLTAN_ID_PTR importGlobalGids{}, importLocalGids{}, exportGlobalGids{}, exportLocalGids{};
    int *importProcs{}, *importToPart{}, *exportProcs{}, *exportToPart{};
};


/** ZoltanGrid is a wrapper for Opm::UnstructuredGrid that provides the neccessarry function
 *  that is required by the Zoltan-domain decomposition library to use perform graph-partitioning
 *  on Opm::UnstructuredGrid.
 *
 *  The intended usage is for the static-functions to be registered as callbacks to Zoltan
 *  and an Opm::UnstructuredGrid (passed via void*) is accepted as the first argument.
 */
class ZoltanGrid {
public:
    static int getNumberOfObjects( void* data, int *ierr );

    static void getCellList( void *data, int sizeGID, int sizeLID,
                             ZOLTAN_ID_PTR globalId, ZOLTAN_ID_PTR localId,
                             int wgt_dim, float *weights, int *ierr );

    static void getNumberOfEdgesMulti( void* data, int num_gid_entries, int num_lid_entries, int num_obj,
                                      ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id,
                                      int* num_edges, int *ierr);

    static void getEdgeListMulti( void *data, int num_gid_entries, int num_lid_entries, int num_obj,
                                  ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids,
                                  int* num_edges, ZOLTAN_ID_PTR nbor_global_id,
                                  int *nbor_procs, int wgt_dim, float *ewgts, int *ierr);

    /** Debug function to dump exports to a stream. */
    static void dumpRank0Exports( const int numCells, const zoltanReturns&, std::ostream& out );    
};


} // namespace equelle

