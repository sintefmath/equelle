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
    createGeneralDescription("CudaMatrix::CudaMatrix()");
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

    createGeneralDescription("CudaMatrix host constructor");
}


// Copy constructor:
CudaMatrix::CudaMatrix(const CudaMatrix& mat)
    : rows_(mat.rows_),
      cols_(mat.cols_),
      nnz_(mat.nnz_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess)
{
    // Copy arrays if they exist:
    if ( mat.csrVal_ != 0 ) {
	cudaStatus_ = cudaMalloc( (void**)&csrVal_, nnz_*sizeof(double));
	checkError_("cudaMalloc(csrVal_) in CudaMatrix copy constructor");
	cudaStatus_ = cudaMemcpy( csrVal_, mat.csrVal_, nnz_*sizeof(double),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrVal_) in CudaMatrix copy constructor");
    }
    if ( mat.csrRowPtr_ != 0 ) {
	cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (rows_+1)*sizeof(int));
	checkError_("cudaMalloc(csrRowPtr_) in CudaMatrix copy constructor");
	cudaStatus_ = cudaMemcpy( csrRowPtr_, mat.csrRowPtr_, (rows_+1)*sizeof(int),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix copy constructor");
    }
    if ( mat.csrColInd_ != 0 ) {
	cudaStatus_ = cudaMalloc( (void**)&csrColInd_, nnz_*sizeof(int));
	checkError_("cudaMalloc(csrColInd_) in CudaMalloc copy constructor");
	cudaStatus_ = cudaMemcpy( csrColInd_, mat.csrColInd_, nnz_*sizeof(int),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrColInd_) in CudaMatrix copy constructor");
    }
    
    createGeneralDescription("CudaMatrix copy constructor");
}


// Copy assignment operator:
CudaMatrix& CudaMatrix::operator= (const CudaMatrix& other) {

    // Protect against self assignment
    if ( this != &other ) {
	
	// Check if we have to reallocate memory depending on nnz:
	if ( nnz_ != other.nnz_ ) { 
	    nnz_ = other.nnz_;
	    
	    // Free and reallocate csrVal_, but only free if csrVal_ != 0
	    if ( csrVal_ != 0 ) {
		cudaStatus_ = cudaFree(csrVal_);
		checkError_("cudaFree(csrVal_) in CudaMatrix copy assignment operator");
	    }
	    cudaStatus_ = cudaMalloc( (void**)&csrVal_, nnz_*sizeof(double));
	    checkError_("cudaMalloc(csrVal_) in CudaMatrix copy assignment operator");
	    
	    // Free (if nonzero) and allocate csrColInd_
	    if ( csrColInd_ != 0 ) {
		cudaStatus_ = cudaFree(csrColInd_);
		checkError_("cudaFree(csrColInd_) in CudaMatrix copy assignment operator");
	    }
	    cudaStatus_ = cudaMalloc( (void**)&csrColInd_, nnz_*sizeof(int));
	    checkError_("cudaMalloc(csrColInd_) in CudaMatrix copy assignment operator");
	} // if (nnz != other.nnz_)

	// Check if we have to reallocate memory depending on rows:
	if ( rows_ != other.rows_ ) {
	    rows_ = other.rows_;
	    if ( csrRowPtr_ != 0 ) {
		cudaStatus_ = cudaFree(csrRowPtr_);
		checkError_("cudaFree(csrRowPtr_) in CudaMatrix copy assignment operator");
	    }
	    cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (rows_+1)*sizeof(int));
	    checkError_("cudaMalloc(csrRowPtr_) in CudaMatrix copy assignment operator");
	} // if ( rows_ != other.rows_ )

	cols_ = other.cols_;
	
	// All arrays correct sizes. Copy data:
	cudaStatus_ = cudaMemcpy( csrVal_, other.csrVal_, nnz_*sizeof(double),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrVal_) in CudaMatrix copy assignment operator");
	cudaStatus_ = cudaMemcpy( csrRowPtr_, other.csrRowPtr_, (rows_+1)*sizeof(int),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix copy assignment operator");
	cudaStatus_ = cudaMemcpy( csrColInd_, other.csrColInd_, nnz_*sizeof(int),
				  cudaMemcpyDeviceToDevice);
	checkError_("cudaMemcpy(csrColInd_) in CudaMatrix copy assignment operator");

	
	// Do not have to care about description, as it is the same for all matrices!
	
    } // if ( this != &other)
    
    return *this;
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

    sparseStatus_ = cusparseDestroyMatDescr( description_ );
    checkError_("cusparseDestroyMatDescr() in CudaMatrix::~CudaMatrix()");

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

void CudaMatrix::createGeneralDescription(const std::string& msg) {
    sparseStatus_ = cusparseCreateMatDescr( &description_ );
    checkError_("cusparseCreateMatDescr() in " + msg);
    
    sparseStatus_ = cusparseSetMatType(description_, CUSPARSE_MATRIX_TYPE_GENERAL);
    checkError_("cusparseSetMatType() in " + msg);
    sparseStatus_ = cusparseSetMatIndexBase(description_, CUSPARSE_INDEX_BASE_ZERO);
    checkError_("cusparseSetMatIndexBase() in " + msg);
}