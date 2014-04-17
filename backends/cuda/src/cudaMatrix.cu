#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <opm/core/utility/ErrorMacros.hpp>

#include <vector>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include "CudaMatrix.hpp"
#include "CudaArray.hpp" // kernels for scalar multiplications
#include "equelleTypedefs.hpp"

using namespace equelleCUDA;
using namespace wrapCudaMatrix;

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
      cudaStatus_(cudaSuccess),
      description_(0)
{
    createGeneralDescription_("CudaMatrix::CudaMatrix()");
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
      cudaStatus_(cudaSuccess),
      description_(0)
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

    createGeneralDescription_("CudaMatrix host constructor");
}


// Constructor from Eigen Matrix
CudaMatrix::CudaMatrix(const Eigen_M& eigen)
    : rows_(eigen.rows()),
      cols_(eigen.cols()),
      nnz_(eigen.nonZeros()),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0)
{
    // Should have a check here to ensure that the matrix is stored 
    // in a row-major format.
    
    // Opm::HelperOps creates helper matrices in column major format.
    // Copy the input to a row major matrix instead:
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> rowmajor(eigen);
    std::cout << "Rowmajor:\n" << rowmajor << "\n";

    // Allocate memory:
    cudaStatus_ = cudaMalloc( (void**)&csrVal_, nnz_*sizeof(double));
    checkError_("cudaMalloc(csrVal_) in CudaMatrix Eigen constructor");
    cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (rows_+1)*sizeof(int));
    checkError_("cudaMalloc(csrRowPtr_) in CudaMatrix Eigen constructor");
    cudaStatus_ = cudaMalloc( (void**)&csrColInd_, nnz_*sizeof(int));
    checkError_("cudaMalloc(csrColInd_) in CudaMatrix Eigen constructor");

    // Copy arrays:
    cudaStatus_ = cudaMemcpy( csrVal_, rowmajor.valuePtr(), nnz_*sizeof(double),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrVal_) in CudaMatrix Eigen constructor");
    cudaStatus_ = cudaMemcpy( csrRowPtr_, rowmajor.outerIndexPtr(), (rows_+1)*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix Eigen constructor");
    cudaStatus_ = cudaMemcpy( csrColInd_, rowmajor.innerIndexPtr(), nnz_*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrColInd_) in CudaMatrix Eigen constructor");

    createGeneralDescription_("CudaMatrix Eigen constructor");
} // constructor from Eigen


// Identity matrix constructor
CudaMatrix::CudaMatrix(const int size) 
    : rows_(size),
      cols_(size),
      nnz_(size),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0)
{
    // Allocate memory:
    cudaStatus_ = cudaMalloc( (void**)&csrVal_, size*sizeof(double));
    checkError_("cudaMalloc(csrVal_) in CudaMatrix identity matrix constructor");
    cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (size+1)*sizeof(int));
    checkError_("cudaMalloc(csrRowPtr_) in CudaMatrix identity matrix constructor");
    cudaStatus_ = cudaMalloc( (void**)&csrColInd_, size*sizeof(int));
    checkError_("cudaMalloc(csrColInd_) in CudaMatrix identity matrix constructor");

    // Call a kernel that writes the correct data:
    kernelSetup s(size+1);
    initIdentityMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_, nnz_);

    createGeneralDescription_("CudaMatrix identity matrix constructor");
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
      cudaStatus_(cudaSuccess),
      description_(0)
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
    
    createGeneralDescription_("CudaMatrix copy constructor");
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

    // Destroy description_ 
    sparseStatus_ = cusparseDestroyMatDescr( description_ );
    checkError_("cusparseDestroyMatDescr() in CudaMatrix::~CudaMatrix()");

    std::cout << "Freeing matrix\n";

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
	OPM_THROW(std::runtime_error, "Tried to copy matrix to host, but the pointers are (" << csrVal_ << "," << csrRowPtr_ << "," << csrColInd_ << ")");
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

void CudaMatrix::createGeneralDescription_(const std::string& msg) {
    sparseStatus_ = cusparseCreateMatDescr( &description_ );
    checkError_("cusparseCreateMatDescr() in " + msg);
    
    sparseStatus_ = cusparseSetMatType(description_, CUSPARSE_MATRIX_TYPE_GENERAL);
    checkError_("cusparseSetMatType() in " + msg);
    sparseStatus_ = cusparseSetMatIndexBase(description_, CUSPARSE_INDEX_BASE_ZERO);
    checkError_("cusparseSetMatIndexBase() in " + msg);
}



// Operator +
CudaMatrix equelleCUDA::operator+(const CudaMatrix& lhs, const CudaMatrix& rhs) {
    return cudaMatrixSum(lhs, rhs, 1.0);
}

CudaMatrix equelleCUDA::operator-(const CudaMatrix& lhs, const CudaMatrix& rhs) {
    return cudaMatrixSum(lhs, rhs, -1.0);
}


CudaMatrix equelleCUDA::cudaMatrixSum(const CudaMatrix& lhs, 
				      const CudaMatrix& rhs,
				      const double beta) {
  
    if ( (lhs.rows_ != rhs.rows_) || (lhs.cols_ != rhs.cols_) ) {
    	OPM_THROW(std::runtime_error, "Error in CudaMatrix + CudaMatrix\n" << "\tMatrices of different size.\n" << "\tlhs: " << lhs.rows_ << " x " << lhs.cols_ << "\n" << "\trhs: " << rhs.rows_ << " x " << rhs.cols_ << ".");
    }

    // Create an empty matrix. Need to set rows, cols, nnz, and allocate arrays!
    CudaMatrix out;
    out.rows_ = lhs.rows_;
    out.cols_ = lhs.cols_;

    // Addition in two steps
    //    1) Find nonzero pattern of output
    //    2) Add matrices.

    // 1) Find nonzero pattern:
    // Allocate rowPtr:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in cudaMatrixSum()");

    // The following code for finding number of non-zeros is
    // taken from the Nvidia cusparse documentation, section 9.1
    // Only additions are the error checking.
    // The documentation has a typo, as it says cusparseSetPointerNode, not Mode.
    int *nnzTotalDevHostPtr = &out.nnz_;
    out.sparseStatus_ = cusparseSetPointerMode( CUSPARSE, CUSPARSE_POINTER_MODE_HOST);
    out.checkError_("cusparseSetPointerMode() in cudaMatrixSum()");
    out.sparseStatus_ = cusparseXcsrgeamNnz( CUSPARSE, out.rows_, out.cols_,
					     lhs.description_, lhs.nnz_,
					     lhs.csrRowPtr_, lhs.csrColInd_,
					     rhs.description_, rhs.nnz_,
					     rhs.csrRowPtr_, rhs.csrColInd_,
					     out.description_, out.csrRowPtr_,
					     nnzTotalDevHostPtr);
    out.checkError_("cusparseXcsrgeamNnz() in cudaMatrixSum()");
    if ( nnzTotalDevHostPtr != NULL) {
	out.nnz_ = *nnzTotalDevHostPtr;
    } else {
	out.cudaStatus_ = cudaMemcpy( &out.nnz_, out.csrRowPtr_ + out.rows_,
				      sizeof(int), cudaMemcpyDeviceToHost);
	out.checkError_("cudaMemcpy(out.csrRowPtr_ + rows_) in cudaMatrixSum()");
	int baseC;
	out.cudaStatus_ = cudaMemcpy( &baseC, out.csrRowPtr_, sizeof(int),
				      cudaMemcpyDeviceToHost);
	out.checkError_("cudaMemcpy(&baseC) in cudaMatrixSum()");
	out.nnz_ -= baseC;
    }

    std::cout << "New nnz = " << out.nnz_ << std::endl;
    
    // Allocate the other two arrays:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrVal_, out.nnz_*sizeof(double));
    out.checkError_("cudaMalloc(out.csrVal_) in cudaMatrixSum()");
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrColInd_, out.nnz_*sizeof(int));
    out.checkError_("cudaMalloc(out.csrColInd_) in cudaMatrixSum()");
    
    // 2) Add matrices
    // Need to create alpha and beta:
    const double alpha = 1.0;
    //double beta = 1.0;
    out.sparseStatus_ = cusparseDcsrgeam(CUSPARSE, out.rows_, out.cols_,
					 &alpha,
					 lhs.description_, lhs.nnz_,
					 lhs.csrVal_, lhs.csrRowPtr_, lhs.csrColInd_,
					 &beta,
					 rhs.description_, rhs.nnz_,
					 rhs.csrVal_, rhs.csrRowPtr_, rhs.csrColInd_,
					 out.description_,
					 out.csrVal_, out.csrRowPtr_, out.csrColInd_);
    out.checkError_("cusparseDcsrgream() in cudaMatrixSum()");

    return out;

} // cudaMatrixSum



CudaMatrix equelleCUDA::operator*(const CudaMatrix& lhs, const CudaMatrix& rhs) {

    if ( lhs.cols_ != rhs.rows_ ) {
	OPM_THROW(std::runtime_error, "Error in CudaMatrix * CudaMatrix\n" << "\tMatrices of illegal sizes.\n" << "\tlhs.cols_ = " << lhs.cols_ << "\n\trhs.rows_ = " << rhs.rows_);
    }

    // Create an empty matrix. Need to set rows, cols, nnz, and allocate arrays!
    CudaMatrix out;
    out.rows_ = lhs.rows_;
    out.cols_ = rhs.cols_;

    // Addition in two steps
    //    1) Find nonzero pattern of output
    //    2) Multiply matrices.

    // 1) Find nonzero pattern of output
    // Allocate rowPtr:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrRowPtr_, (out.rows_+1)*sizeof(int));
    out.checkError_("cudaMalloc(out.csrRowPtr_) in CudaMatrix operator +");

    // The following code for finding number of non-zeros is
    // taken from the Nvidia cusparse documentation, section 9.2
    // Only additions are the error checking.
    int *nnzTotalDevHostPtr = &out.nnz_;
    out.sparseStatus_ = cusparseSetPointerMode(CUSPARSE, CUSPARSE_POINTER_MODE_HOST);
    out.checkError_("cusparseSetPointerMode() in CudaMatrix operator *");
    out.sparseStatus_ = cusparseXcsrgemmNnz( CUSPARSE, 
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     CUSPARSE_OPERATION_NON_TRANSPOSE,
					     out.rows_, out.cols_, lhs.cols_,
					     lhs.description_, lhs.nnz_,
					     lhs.csrRowPtr_, lhs.csrColInd_,
					     rhs.description_, rhs.nnz_,
					     rhs.csrRowPtr_, rhs.csrColInd_,
					     out.description_,
					     out.csrRowPtr_, nnzTotalDevHostPtr);
    out.checkError_("cusparseXcsrgemmNnz() in CudaMatrix operator *");
    if ( nnzTotalDevHostPtr != NULL ) {
	out.nnz_ = *nnzTotalDevHostPtr;
    } else {
	int baseC;
	out.cudaStatus_ = cudaMemcpy(&out.nnz_, out.csrRowPtr_ + out.rows_,
				     sizeof(int), cudaMemcpyDeviceToHost);
	out.checkError_("cudaMemcpy(out.csrRowPtr_ + out.rows_) in CudaMatrix operator *");
	out.cudaStatus_ = cudaMemcpy(&baseC, out.csrRowPtr_, sizeof(int),
				     cudaMemcpyDeviceToHost);
	out.checkError_("cudaMemcpy(baseC) in CudaMatrix operator *");
	out.nnz_ -= baseC;
    }

    std::cout << "New nnz: " << out.nnz_ << "\n";
    
    // Allocate the other two arrays:
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrVal_, out.nnz_*sizeof(double));
    out.checkError_("cudaMalloc(out.csrVal_) in CudaMatrix operator *");
    out.cudaStatus_ = cudaMalloc( (void**)&out.csrColInd_, out.nnz_*sizeof(int));
    out.checkError_("cudaMalloc(out.csrColInd_) in CudaMatrix operator *");
    
    // 2) Multiply the matrices:
    out.sparseStatus_ = cusparseDcsrgemm(CUSPARSE,
					 CUSPARSE_OPERATION_NON_TRANSPOSE,
					 CUSPARSE_OPERATION_NON_TRANSPOSE,
					 out.rows_, out.cols_, lhs.cols_,
					 lhs.description_, lhs.nnz_,
					 lhs.csrVal_, lhs.csrRowPtr_, lhs.csrColInd_,
					 rhs.description_, rhs.nnz_,
					 rhs.csrVal_, rhs.csrRowPtr_, rhs.csrColInd_,
					 out.description_,
					 out.csrVal_, out.csrRowPtr_, out.csrColInd_);
    out.checkError_("cusparseDcsrgemm() in CudaMatrix operator *");
    
    return out;
} // operator *

// Scalar multiplications with matrix:
CudaMatrix equelleCUDA::operator*(const CudaMatrix& lhs, const Scalar rhs) {
    CudaMatrix out(lhs);
    kernelSetup s(out.nnz_);
    wrapCudaArray::scalMultColl_kernel<<<s.grid, s.block>>>(out.csrVal_,
							    rhs,
							    out.nnz_);
    return out;
}

CudaMatrix equelleCUDA::operator*(const Scalar lhs, const CudaMatrix& rhs) {
    CudaMatrix out(rhs);
    kernelSetup s(out.nnz_);
    wrapCudaArray::scalMultColl_kernel<<<s.grid, s.block>>>(out.csrVal_,
							    lhs,
							    out.nnz_);
    return out;
}


__global__ void wrapCudaMatrix::initIdentityMatrix(double* csrVal,
						   int* csrRowPtr,
						   int* csrColInd,
						   const int nnz)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if ( i < nnz + 1) {
	csrRowPtr[i] = i;
	if (i < nnz) {
	    csrVal[i] = 1.0;
	    csrColInd[i] = i;
	}
    }
}