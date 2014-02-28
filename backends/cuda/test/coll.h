
// coll.h

#ifndef EQUELLE_CUDA_TEST_COLL_HEADER
#define EQUELLE_CUDA_TEST_COLL_HEADER


#include <cuda.h>
#include <cuda_runtime.h>
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"


// Function declarations:

void test_copy(equelleCUDA::CollOfFaces coll);
void test_full(equelleCUDA::CollOfCells coll);
void test_back_to_host(equelleCUDA::CollOfCells coll);

int cuda_main();


#endif // EQUELLE_CUDA_TEST_COLL_HEADER
