#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <opm/core/utility/ErrorMacros.hpp>

#include <vector>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include "CudaMatrix.hpp"
#include "CudaArray.hpp" // kernels for scalar multiplications
#include "CollOfScalar.hpp" // for constructor for diagonal matrix.
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
    // Allocate memory
    allocateMemory("CudaMatrix host constructor");

    // Copy data:
    cudaStatus_ = cudaMemcpy( csrVal_, val, nnz_*sizeof(double), 
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrVal_) in CudaMatrix host constructor");

    cudaStatus_ = cudaMemcpy( csrRowPtr_, rowPtr, (rows_ + 1)*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrRowPtr_) in CudaMatrix host constructor");

    cudaStatus_ = cudaMemcpy( csrColInd_, colInd, nnz_*sizeof(int),
			      cudaMemcpyHostToDevice);
    checkError_("cudaMemcpy(csrColInd_) in CudaMatrix host constructor");

    createGeneralDescription_("CudaMatrix host constructor");
}


// Constructor from Eigen Matrix
CudaMatrix::CudaMatrix(const Eigen::SparseMatrix<Scalar>& eigen)
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

    // Allocate memory:
    allocateMemory("CudaMatrix Eigen constructor");

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
    allocateMemory("CudaMatrix identity matrix constructor");

    // Call a kernel that writes the correct data:
    kernelSetup s(size+1);
    initIdentityMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_, nnz_);

    createGeneralDescription_("CudaMatrix identity matrix constructor");
}



// Constructor for creating a diagonal matrix from the value of a CollOfScalar
CudaMatrix::CudaMatrix(const CollOfScalar& coll)
    : rows_(coll.size()),
      cols_(rows_),
      nnz_(rows_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0)
{
    // Allocate memory:
    allocateMemory("CudaMatrix diagonal matrix constructor");

    // Call a kernel to write the correct data:
    kernelSetup s(nnz_+1);
    initDiagonalMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_, coll.data(),
					    nnz_);
    
    createGeneralDescription_("CudaMatrix diagonal matrix constructor");
}

// Constructor for creating a diagonal matrcit from a CudaArray
CudaMatrix::CudaMatrix(const CudaArray& array)
    : rows_(array.size()),
      cols_(rows_),
      nnz_(rows_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0)
{
    // Allocate memory:
    allocateMemory("CudaMatrix::CudaMatrix(CudaArray)");

    // Call a kerenl to write the correct data:
    kernelSetup s(nnz_ + 1);
    initDiagonalMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_,
					    array.data(), nnz_);
    
    createGeneralDescription_("CudaMatrix::CudaMatrix(CudaArray)");
}
					    

// Restriction matrix constructor:
CudaMatrix::CudaMatrix(const thrust::device_vector<int> set,
		       const int full_size) 
    : rows_(set.size()),
      cols_(full_size),
      nnz_(rows_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0)
{
    // Allocate memory:
    allocateMemory("CudaMatrix constructor for On from full set");
    
    // Matrix is flat, more cols than rows.
    //   - each row has one element, hence csrRowPtr = [0,1,2,...,rows_] (size rows+1)
    //   - all nnz elements are 1, hence csrVal = [1,1,1,...,1] (size rows)
    //   - csrColInd = to_set (size rows)
    const int* set_ptr = thrust::raw_pointer_cast( &set[0] );
    kernelSetup s(rows_ + 1);
    initRestrictionMatrix<<<s.grid, s.block>>>( csrVal_, csrRowPtr_, csrColInd_,
						set_ptr, rows_);

    createGeneralDescription_("CudaMatrix constructor for On from full set");
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
	
	if ( !other.isEmpty() ) {
	    
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
	    
	} // other is not empty
	else {
	    // Other is empty!
	    if ( !this->isEmpty() ) {
		if ( csrVal_ != 0 ) {
		    cudaStatus_ = cudaFree(csrVal_);
		    checkError_("cudaFree(csrVal_) in CudaMatrix copy assignment for empty other");
		    csrVal_ = 0;
		}
		if ( csrRowPtr_ != 0 ) {
		    cudaStatus_ = cudaFree(csrRowPtr_);
		    checkError_("cudaFree(csrRowPtr_) in CudaMatrix copy assignment for empty other");
		    csrRowPtr_ = 0;
		}
		if ( csrColInd_ != 0 ) {
		    cudaStatus_ = cudaFree(csrColInd_);
		    checkError_("cudaFree(csrColInd_) in CudaMatrix copy assignment for empty other");
		    csrColInd_ = 0;
		}
		nnz_ = 0;
		rows_ = 0;
		cols_ = 0;
	    }
       

	} // if other is empty
	
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

bool CudaMatrix::isEmpty() const {
    return (csrVal_ == NULL);
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

void CudaMatrix::checkError_(const std::string& msg, const std::string& caller) const {
    checkError_(msg + caller);
}

void CudaMatrix::createGeneralDescription_(const std::string& msg) {
    sparseStatus_ = cusparseCreateMatDescr( &description_ );
    checkError_("cusparseCreateMatDescr() in " + msg);
    
    sparseStatus_ = cusparseSetMatType(description_, CUSPARSE_MATRIX_TYPE_GENERAL);
    checkError_("cusparseSetMatType() in " + msg);
    sparseStatus_ = cusparseSetMatIndexBase(description_, CUSPARSE_INDEX_BASE_ZERO);
    checkError_("cusparseSetMatIndexBase() in " + msg);
}


// MEMORY ALLOCATIONS
void CudaMatrix::allocateMemory(const std::string& caller) {
    // Error checking:
    if ( csrVal_ != 0 ) 
	OPM_THROW(std::runtime_error, "Error in CudaMatrix::allocateMemory\n" << "\tcsrVal_ already allocated.\n\tCalled from " << caller);
    if ( csrRowPtr_ != 0 ) {
	OPM_THROW(std::runtime_error, "Error in CudaMatrix::allocateMemory\n" << "\tcsrRowPtr_ already allocated.\n\tCalled from " << caller);
    }
    if ( csrColInd_ != 0 ) {
	OPM_THROW(std::runtime_error, "Error in CudaMatrix::allocateMemory\n" << "\tcsrColInd_ already allocated.\n\tCalled from " << caller);
    }
    
    // Allocating
    cudaStatus_ = cudaMalloc( (void**)&csrVal_, nnz_*sizeof(double));
    checkError_("cudaMalloc(csrVal_) in ", caller);
    cudaStatus_ = cudaMalloc( (void**)&csrRowPtr_, (rows_+1)*sizeof(int));
    checkError_("cudaMalloc(csrRowPtr_) in ", caller);
    cudaStatus_ = cudaMalloc( (void**)&csrColInd_, nnz_*sizeof(int));
    checkError_("cudaMalloc(csrColInd_) in ", caller);
}


// --------------------- OVERLOADING OF OPERATORS -------------------------- //

// Operator +
CudaMatrix equelleCUDA::operator+(const CudaMatrix& lhs, const CudaMatrix& rhs) {
    // If one of the matrices is emtpy, we interpret it as a matrix filled with
    // zeros, and therefore just return the other matrix.
    // This is convenient when we implement autodiff by using CudaMatrix.
    if ( lhs.isEmpty() ) {
	return rhs;
    } 
    else if ( rhs.isEmpty() ) {
	return lhs;
    } 
    else {
	std::cout << "Matrix plus!\n";
	return cudaMatrixSum(lhs, rhs, 1.0);
    }
}

CudaMatrix equelleCUDA::operator-(const CudaMatrix& lhs, const CudaMatrix& rhs) {
    // If one of the matrices is emtpy, we interpret it as a matrix filled with
    // zeros, and therefore just return the other matrix.
    // This is convenient when we implement autodiff by using CudaMatrix.
    if ( lhs.isEmpty() ) {
	return -1.0*rhs;
    }
    else if ( rhs.isEmpty() ) {
	return lhs;
    }
    else {
	return cudaMatrixSum(lhs, rhs, -1.0);
    }
}


CudaMatrix equelleCUDA::cudaMatrixSum(const CudaMatrix& lhs, 
				      const CudaMatrix& rhs,
				      const double beta) {
  
    if ( lhs.isEmpty() || rhs.isEmpty() ) {
	if ( lhs.isEmpty() ) 
	    OPM_THROW(std::runtime_error, "Calling cudaMatrixSum with lhs empty");
	else 
	    OPM_THROW(std::runtime_error, "Calling cudaMatrixSum with rhs empty");
    }

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

    // If any of them are empty, we return an empty matrix.
    // An empty matrix is interpreted as a correctly sized matrix of zeros.
    // This lets us not worry about empty derivatives for autodiff.
    if ( lhs.isEmpty() || rhs.isEmpty() ) {
	return CudaMatrix();
    }
    
    if ( lhs.cols_ != rhs.rows_ ) {
	OPM_THROW(std::runtime_error, "Error in CudaMatrix * CudaMatrix\n" << "\tMatrices of illegal sizes.\n" << "\tlhs.cols_ = " << lhs.cols_ << "\n\trhs.rows_ = " << rhs.rows_);
    }
    std::cout << "Matrix mult\n";

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


// Matrix * vector
CudaArray equelleCUDA::operator*(const CudaMatrix& mat, const CudaArray& vec) {
    // Check that sizes match
    if ( mat.cols_ != vec.size() ) {
	OPM_THROW(std::runtime_error, "Error in matrix * vector operation as matrix is of size " << mat.rows_ << " by " << mat.cols_ << " and the vector of size " << vec.size());
    }
    
    // Call cusparse matrix-vector operation:
    // y = alpha*op(A)*x + beta*y
    // with alpha=1, beta=0, op=non_transpose
    CudaArray out(mat.rows());
    const double alpha = 1.0;
    const double beta = 0.0;
    mat.sparseStatus_ = cusparseDcsrmv( CUSPARSE,
					CUSPARSE_OPERATION_NON_TRANSPOSE,
					mat.rows_, mat.cols_, mat.nnz_, 
					&alpha, mat.description_,
					mat.csrVal_, mat.csrRowPtr_, mat.csrColInd_,
					vec.data(), &beta,
					out.data());
    mat.checkError_("cusparseDcsrmv() in operator*(CudaMatrix, CudaArray)");
    return out;
}



// Scalar multiplications with matrix:
CudaMatrix equelleCUDA::operator*(const CudaMatrix& lhs, const Scalar rhs) {
    return (rhs * lhs);
}

CudaMatrix equelleCUDA::operator*(const Scalar lhs, const CudaMatrix& rhs) {
    // rhs should not be empty
    if ( rhs.isEmpty() ) {
	OPM_THROW(std::runtime_error, "Calling CudaMatrix * Scalar with empty matrix...");
    }
    
    CudaMatrix out(rhs);
    kernelSetup s(out.nnz_);
    wrapCudaArray::scalMultColl_kernel<<<s.grid, s.block>>>(out.csrVal_,
							    lhs,
							    out.nnz_);
    return out;
}

CudaMatrix equelleCUDA::operator-(const CudaMatrix& arg) {
    return -1.0*arg;
}



// KERNELS -------------------------------------------------


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


__global__ void wrapCudaMatrix::initDiagonalMatrix( double* csrVal,
						    int* csrRowPtr,
						    int* csrColInd,
						    const double* scalars,
						    const int nnz)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if ( i < nnz + 1) {
	csrRowPtr[i] = i;
	if ( i < nnz) {
	    csrVal[i] = scalars[i];
	    csrColInd[i] = i;
	}
    }
}


// Restriction matrix initialization kernel
__global__ void wrapCudaMatrix::initRestrictionMatrix( double* csrVal,
						       int* csrRowPtr,
						       int* csrColInd,
						       const int* set,
						       const int rows) {
    // Matrix is flat, more cols than rows.
    //   - each row has one element, hence csrRowPtr = [0,1,2,...,rows_] (size rows+1)
    //   - all nnz elements are 1, hence csrVal = [1,1,1,...,1] (size rows)
    //   - csrColInd = to_set (size rows)
    const int i = threadIdx.x + blockIdx.x*blockDim.x;
    if ( i < rows + 1) {
	csrRowPtr[i] = i;
	if ( i < rows ) {
	    csrVal[i] = 1;
	    csrColInd[i] = set[i];
	}
    }
}