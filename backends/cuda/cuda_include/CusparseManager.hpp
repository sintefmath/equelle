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
    static CudaMatrix matrixMultiply(const CudaMatrix& lhs, const CudaMatrix& rhs);
    static CudaMatrix matrixMultiply2(const CudaMatrix& lhs, const CudaMatrix& rhs);
private:
    CusparseManager();
    ~CusparseManager();

    static CusparseManager& instance();
    CudaMatrix gemm(const CudaMatrix& lhs, const CudaMatrix& rhs);
    CudaMatrix gemm2(const CudaMatrix& A, const CudaMatrix& B, const CudaMatrix& C, const double* alpha, const double* beta);

    // cuSPARSE  and CUDA variables
    cusparseHandle_t cusparseHandle_;
    cusparseStatus_t sparseStatus_;
    csrgemm2Info_t gemm2Info_;
    cudaError_t cudaStatus_;
    void* buffer_;
    size_t currentBufferSize_;
};
}

#endif