
// gridTest.cu

#ifndef EQUELLE_CUDA_TEST_GRIDTEST_HEADER
#define EQUELLE_CUDA_TEST_GRIDTEST_HEADER

#include "deviceGrid.hpp"

int all_cells_test(equelleCUDA::DeviceGrid dg);
int all_faces_test(equelleCUDA::DeviceGrid dg);
int boundary_faces_test(equelleCUDA::DeviceGrid dg);
int interior_faces_test(equelleCUDA::DeviceGrid dg);
int boundary_cells_test(equelleCUDA::DeviceGrid dg);

int cuda_main(equelleCUDA::DeviceGrid dg);


#endif // EQUELLE_CUDA_TEST_GRIDTEST_HEADER

