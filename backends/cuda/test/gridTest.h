
// gridTest.cu

#ifndef EQUELLE_CUDA_TEST_GRIDTEST_HEADER
#define EQUELLE_CUDA_TEST_GRIDTEST_HEADER

#include <string>

#include "DeviceGrid.hpp"

int compare_collection(equelleCUDA::CollOfIndices coll,
		       int sol[], int sol_size, bool full,
		       std::string test);


int cuda_main(equelleCUDA::DeviceGrid dg);


#endif // EQUELLE_CUDA_TEST_GRIDTEST_HEADER

