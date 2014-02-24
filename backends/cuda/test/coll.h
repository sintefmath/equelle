
// coll.h

#ifndef EQUELLE_CUDA_TEST_COLL_HEADER
#define EQUELLE_CUDA_TEST_COLL_HEADER


#include <cuda.h>
#include <cuda_runtime.h>
#include <deviceGrid.hpp>


// Function declarations:

void test_copy(equelleCUDA::Collection coll);
void test_full(equelleCUDA::Collection coll);
void test_back_to_host(equelleCUDA::Collection coll);

int cuda_main();


#endif // EQUELLE_CUDA_TEST_COLL_HEADER
