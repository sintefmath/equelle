#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <opm/core/utility/ErrorMacros.hpp>

#include <vector>
#include <iostream>
#include <string>

#include "CudaMatrix.hpp"


using namespace equelleCUDA;
using std::vector;

// Implementation of member functions of CudaMatrix

// Default constructor:
CudaMatrix::CudaMatrix() 
    : rows_(0),
      cols_(0),
      nnz_(0),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess)
{
    // Intentionally left blank
}


// Constructor from host values:
CudaMatrix::CudaMatrix( const double* val, const int* rowPtr, const int* colInd,
			const int nnz, const int rows, const int cols)
    : rows_(rows),
      cols_(cols),
      nnz_(nnz),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess)
{
    // Allocate memory and copy data to host
    cudaStatus_ = cudaMalloc( (void**)&csrVal_, nnz_*sizeof(double));
    checkError_("cudaMalloc(csrVal_) in CudaMatrix host constructor");
    cudaStatus_ = cudaMemcpy( csrVal_, val, nnz_*sizeof(double), 
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrVal_) in CudaMatrix host constructor");

    cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (rows_ + 1)*sizeof(int));
    checkError_("cudaMalloc(csrRowPtr_) in CudaMatrix host constructor");
    cudaStatus_ = cudaMemcpy( csrRowPtr_, rowPtr, (rows_ + 1)*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix host constructor");

    cudaStatus_ = cudaMalloc( (void**)&csrColInd_, nnz_*sizeof(int));
    checkError_("cudaMalloc(csrColInd_) in CudaMatrix host constructor");
    cudaStatus_ = cudaMemcpy( csrColInd_, colInd, nnz_*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrColInd_) in CudaMatrix host constructor");
}


// Destructor
CudaMatrix::~CudaMatrix() {
    // Free pointers if not zero.
    if ( csrVal_ != 0 ) {
	cudaStatus_ = cudaFree(csrVal_);
	checkError_("cudaFree(csrVal_) in CudaMatrix::~CudaMatrix");
    }
    if ( csrRowPtr_ != 0 ) {
	cudaStatus_ = cudaFree(csrRowPtr_);
	checkError_("cudaFree(csrRowPtr_) in CudaMatrix::~CudaMatrix");
    }
    if ( csrColInd_ != 0 ) {
	cudaStatus_ = cudaFree(csrColInd_);
	checkError_("cudaFree(csrColInd_) in CudaMatrix::~CudaMatrix");
    }
}

int CudaMatrix::nnz() const {
    return nnz_;
}
int CudaMatrix::rows() const {
    return rows_;
}
int CudaMatrix::cols() const {
    return cols_;
}


// Copy to host:
hostMat CudaMatrix::toHost() const {
    if ( (csrVal_ == 0) || (csrRowPtr_ == 0) || (csrColInd_ == 0) ) {
	OPM_THROW(std::runtime_error, "Tried to copy matrix to host, but the pointers are (" << csrVal_ << "," << csrRowPtr_ << "," << csrColInd_ );
    }

    vector<double> v(nnz_, -1);
    cudaStatus_ = cudaMemcpy( &v[0], csrVal_, nnz_*sizeof(double),
			      cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy(csrVal_) in CudaMatrix::toHost()");
    
    vector<int> rp(rows_+1, -1);
    cudaStatus_ = cudaMemcpy( &rp[0], csrRowPtr_, (rows_+1)*sizeof(int),
			      cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix::toHost()");
    
    vector<int> ci(nnz_, -1);
    cudaStatus_ = cudaMemcpy( &ci[0], csrColInd_, nnz_*sizeof(int),
			      cudaMemcpyDeviceToHost);
    checkError_("cudaMemcpy(csrColInd_) in CudaMatrix::toHost()");
    
    hostMat out;
    out.vals = v;
    out.rowPtr = rp;
    out.colInd = ci;
    out.nnz = nnz_;
    out.rows = rows_;
    out.cols = cols_;
    return out;
}


// Error checking:
void CudaMatrix::checkError_(const std::string& msg) const {
    if ( cudaStatus_ != cudaSuccess) {
	OPM_THROW(std::runtime_error, "\nCuda error\n\t" << msg << " - Error code: "<< cudaGetErrorString(cudaStatus_) );
    }
    if ( sparseStatus_ != CUSPARSE_STATUS_SUCCESS ) {
	OPM_THROW(std::runtime_error, "\ncusparse error\n\t" << msg << " - Error code: " << sparseStatus_);
    }
}