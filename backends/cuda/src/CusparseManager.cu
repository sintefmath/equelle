#include "CusparseManager.hpp"

using namespace equelleCUDA;

CusparseManager::CusparseManager()
    : buffer_(NULL),
      currentBufferSize_(0)
{
    //std::cout << "CusparseManager constructed." << std::endl;
    // Set up cuSPARSE
    cusparseCreate(&cusparseHandle_);
    cusparseSetPointerMode(cusparseHandle_, CUSPARSE_POINTER_MODE_HOST);
    cusparseCreateCsrgemm2Info(&gemm2Info_);
}


CusparseManager::~CusparseManager()
{
    //std::cout << "CusparseManager destroyed." << std::endl;
    if (buffer_ != NULL) {
        cudaFree(buffer_);
    }
    cusparseDestroy(cusparseHandle_);
    cusparseDestroyCsrgemm2Info(gemm2Info_);
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

CudaMatrix CusparseManager::matrixMultiply2(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    double alpha = 1.0;
    return instance().gemm2(lhs, rhs, CudaMatrix(), &alpha, NULL);
}


CudaMatrix CusparseManager::gemm(const CudaMatrix& lhs, const CudaMatrix& rhs)
{

    // Declare output matrix and set its dimensions (if lhs and rhs are compatible).
    CudaMatrix out;
    int innerSize = out.confirmMultSize(lhs, rhs);

    // Allocate row pointer array.
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in CusparseManager::gemm()");
    
    // Find the resulting non-zero pattern.
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


CudaMatrix CusparseManager::gemm2(const CudaMatrix& A, const CudaMatrix& B, const CudaMatrix& C, const double* alpha, const double* beta)
{
    // Declare output matrix and set its dimensions (if lhs and rhs are compatible).
    CudaMatrix out;
    int innerSize = out.confirmMultSize(A, B);

    // Allocate row pointer array.
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc((void**)&out.csrRowPtr_, sizeof(int)*(out.rows_+1)) in CusparseManager::gemm2()");

    // Compute needed buffer size
    size_t newBufferSize;
    out.sparseStatus_ = cusparseDcsrgemm2_bufferSizeExt(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha,
                                     A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                                     B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                                     beta,
                                     C.description_, C.nnz_, C.csrRowPtr_, C.csrColInd_,
                                     gemm2Info_, &newBufferSize);
    out.checkError_("cusparseDcsrgemm2_bufferSizeExt() in CusparseManager::gemm2()");
    
    // (Re)allocate buffer if needed
    if (newBufferSize > currentBufferSize_) {
        if (buffer_ != NULL) {
            out.cudaStatus_ = cudaFree(buffer_);
            out.checkError_("Error when freeing buffer in CusparseManager::gemm2()");
        }
        out.cudaStatus_ = cudaMalloc(&buffer_, newBufferSize);
        out.checkError_("cudaMalloc(&buffer_, newBufferSize) in CusparseManager::gemm2()");
        currentBufferSize_ = newBufferSize;
    }

    // Find the resulting non-zero pattern.
    out.sparseStatus_ = cusparseXcsrgemm2Nnz(cusparseHandle_,
                         out.rows_, out.cols_, innerSize,
                         A.description_, A.nnz_, A.csrRowPtr_, A.csrColInd_,
                         B.description_, B.nnz_, B.csrRowPtr_, B.csrColInd_,
                         C.description_, C.nnz_, C.csrRowPtr_, C.csrColInd_,
                         out.description_, out.csrRowPtr_,
                         &out.nnz_, gemm2Info_, buffer_);
    out.checkError_("cusparseXcsrgemm2Nnz() in CusparseManager::gemm2()");

    // Allocate value array and column index array.
    out.cudaStatus_ = cudaMalloc((void**)&out.csrColInd_, sizeof(int)*out.nnz_);
    out.checkError_("cudaMalloc((void**)&out.csrColInd_, sizeof(int)*out.nnz_) in CusparseManager::gemm2()");
    out.cudaStatus_ = cudaMalloc((void**)&out.csrVal_, sizeof(double)*out.nnz_);
    out.checkError_("cudaMalloc((void**)&out.csrVal_, sizeof(double)*out.nnz_) in CusparseManager::gemm2()");
    
    // Perform the gemm2 operation
    // out = alpha ∗ A ∗ B + beta ∗ C
    out.sparseStatus_ = cusparseDcsrgemm2(cusparseHandle_, out.rows_, out.cols_, innerSize, alpha, 
                                      A.description_, A.nnz_, A.csrVal_, A.csrRowPtr_, A.csrColInd_, 
                                      B.description_, B.nnz_, B.csrVal_, B.csrRowPtr_, B.csrColInd_,
                                      beta,
                                      C.description_, C.nnz_, C.csrVal_, C.csrRowPtr_, C.csrColInd_,
                                      out.description_, out.csrVal_, out.csrRowPtr_, out.csrColInd_,
                                      gemm2Info_, buffer_);
    out.checkError_("cusparseDcsrgemm2() in CusparseManager::gemm2()");
    return out;
}