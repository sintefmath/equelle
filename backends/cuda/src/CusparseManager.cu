#include "CusparseManager.hpp"

using namespace equelleCUDA;

CusparseManager::CusparseManager()
{
    //std::cout << "CusparseManager constructed." << std::endl;
    // Set up cuSPARSE
    cusparseCreate(&cusparseHandle_);
    cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
}

CusparseManager::~CusparseManager()
{
    //std::cout << "CusparseManager destroyed." << std::endl;
    cusparseDestroy(cusparseHandle_);
}

/// Using the Meyers singleton pattern.
CusparseManager& CusparseManager::instance()
{
    static CusparseManager s;
    return s;
}

CudaMatrix CusparseManager::matrixMultiply(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    return instance().gemm(lhs, rhs);
}

CudaMatrix CusparseManager::gemm(const CudaMatrix& lhs, const CudaMatrix& rhs)
{

    // Declare output matrix and set its dimensions (if lhs and rhs are compatible).
    CudaMatrix out;
    int innerSize = out.confirmMultSize(lhs, rhs);

    // Allocate row pointer array.
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in CusparseManager::gemm()");
    
    // Find the resulting non-zero pattern
    out.sparseStatus_ = cusparseXcsrgemmNnz( cusparseHandle_, 
                         lhs.operation_, rhs.operation_,
                         out.rows_, out.cols_, innerSize,
                         lhs.description_, lhs.nnz_,
                         lhs.csrRowPtr_, lhs.csrColInd_,
                         rhs.description_, rhs.nnz_,
                         rhs.csrRowPtr_, rhs.csrColInd_,
                         out.description_,
                         out.csrRowPtr_, &out.nnz_);
    out.checkError_("cusparseXcsrgemmNnz() in CusparseManager::gemm()");

    // Allocate value array and column index array.
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrVal_, out.nnz_*sizeof(double));
    out.checkError_("cudaMalloc(out.csrVal_) in CusparseManager::gemm()");
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrColInd_, out.nnz_*sizeof(int));
    out.checkError_("cudaMalloc(out.csrColInd_) in CusparseManager::gemm()");

    // Perform the multiplication.
    out.sparseStatus_ = cusparseDcsrgemm(cusparseHandle_,
                     lhs.operation_, rhs.operation_,
                     out.rows_, out.cols_, innerSize,
                     lhs.description_, lhs.nnz_,
                     lhs.csrVal_, lhs.csrRowPtr_, lhs.csrColInd_,
                     rhs.description_, rhs.nnz_,
                     rhs.csrVal_, rhs.csrRowPtr_, rhs.csrColInd_,
                     out.description_,
                     out.csrVal_, out.csrRowPtr_, out.csrColInd_);
    out.checkError_("cusparseDcsrgemm() in CusparseManager::gemm()");
    
    return out;
}