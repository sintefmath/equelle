
// gridTest.cu

#ifndef EQUELLE_CUDA_TEST_GRIDTEST_HEADER
#define EQUELLE_CUDA_TEST_GRIDTEST_HEADER

#include <string>

#include "DeviceGrid.hpp"
#include "equelleTypedefs.hpp"

template <int dummy>
int compare_collection(equelleCUDA::CollOfIndices<dummy> coll,
		       int sol[], int sol_size, bool full,
		       std::string test);

int compare_collection_bool(CollOfBool coll, bool sol[], int size,
			    std::string test);

int cuda_main(equelleCUDA::DeviceGrid dg);


#endif // EQUELLE_CUDA_TEST_GRIDTEST_HEADER

