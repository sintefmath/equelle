#include "equelle/ZoltanGridMigrator.hpp"

int equelle::ZoltanGridMigrator::cellSize(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr)
{
    *ierr = ZOLTAN_FATAL;
}

void equelle::ZoltanGridMigrator::packCell(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int dest, int size, char *buf, int *ierr)
{
    *ierr = ZOLTAN_FATAL;
}

void equelle::ZoltanGridMigrator::unpackCell(void *data, int num_gid_entries, ZOLTAN_ID_PTR global_id, int size, char *buf, int *ierr)
{
    *ierr = ZOLTAN_FATAL;
}


