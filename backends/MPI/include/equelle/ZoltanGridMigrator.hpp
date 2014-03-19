#pragma once

#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <zoltan_cpp.h>
#pragma GCC diagnostic pop

namespace equelle {

class ZoltanGridMigrator {
public:

    static int cellSize(  void *data,
                          int num_gid_entries,
                          int num_lid_entries,
                          ZOLTAN_ID_PTR global_id,
                          ZOLTAN_ID_PTR local_id,
                          int *ierr );

    static void packCell( void *data,  int num_gid_entries,
                          int num_lid_entries,
                          ZOLTAN_ID_PTR global_id,
                          ZOLTAN_ID_PTR local_id,
                          int dest,
                          int size,
                          char *buf,
                          int *ierr );

    static void unpackCell( void *data,
                            int num_gid_entries,
                            ZOLTAN_ID_PTR global_id,
                            int size,
                            char *buf,
                            int *ierr );
};

} // namespace equelle
