#ifndef EQUELLE_CUSPARSE_MANAGER_HEADER_INCLUDED
#define EQUELLE_CUSPARSE_MANAGER_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "CudaMatrix.hpp"

namespace equelleCUDA
{
class CudaMatrix;
class CusparseManager
{
public:
    static CudaMatrix matrixMultiply(const CudaMatrix& A, const CudaMatrix& B);
private:
    CusparseManager();
    ~CusparseManager();

    CudaMatrix gemm(const CudaMatrix& lhs, const CudaMatrix& rhs);
    static CusparseManager& instance();

    // cuSPARSE  and CUDA variables
    cusparseHandle_t cusparseHandle_;
    cusparseStatus_t sparseStatus_;
    cudaError_t cudaStatus_;
};
}

#endif