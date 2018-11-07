#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>
#include <thrust/detail/raw_pointer_cast.h>

#include <opm/common/ErrorMacros.hpp>

#include <vector>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include "CudaMatrix.hpp"
#include "CudaArray.hpp" // kernels for scalar multiplications
#include "CollOfScalar.hpp" // for constructor for diagonal matrix.
#include "equelleTypedefs.hpp"
#include "device_functions.cuh"
#include "CusparseManager.hpp"

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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(false)
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(false)
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(false)
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(true)
{
    // Allocate memory:
    allocateMemory("CudaMatrix identity matrix constructor");

    // Call a kernel that writes the correct data:
    kernelSetup s(size+1);
    initIdentityMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_, nnz_);

    createGeneralDescription_("CudaMatrix identity matrix constructor");
}

// Constructs a matrix of size rows*cols with nnz non-zero elements.
// The constructor allocates memory, but does not initialize it.
CudaMatrix::CudaMatrix(const int rows, const int cols, const int nnz)
    : rows_(rows),
      cols_(cols),
      nnz_(nnz),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(false)
{
    allocateMemory("CudaMatrix(rows, cols, nnz)");

    createGeneralDescription_("CudaMatrix(rows, cols, nnz)");
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(true)
{
    // Allocate memory:
    allocateMemory("CudaMatrix diagonal matrix constructor");

    // Call a kernel to write the correct data:
    kernelSetup s(nnz_+1);
    initDiagonalMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_, coll.data(),
              nnz_);
    
    createGeneralDescription_("CudaMatrix diagonal matrix constructor");
}

// Move constructor
CudaMatrix::CudaMatrix(CudaMatrix&& mat)
    : rows_(mat.rows_),
      cols_(mat.cols_),
      nnz_(mat.nnz_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0),
      operation_(mat.operation_),
      diagonal_(mat.diagonal_)
{
    swap(mat);
    createGeneralDescription_("CudaMatrix move constructor");
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(true)
{
    // Allocate memory:
    allocateMemory("CudaMatrix::CudaMatrix(CudaArray)");

    // Call a kerenl to write the correct data:
    kernelSetup s(nnz_ + 1);
    initDiagonalMatrix<<<s.grid, s.block>>>(csrVal_, csrRowPtr_, csrColInd_,
              array.data(), nnz_);
    
    createGeneralDescription_("CudaMatrix::CudaMatrix(CudaArray)");
}
              

// Constructor for diagonal from booleans
CudaMatrix::CudaMatrix(const CollOfBool& bools) 
    : rows_(bools.size()),
      cols_(rows_),
      nnz_(rows_),
      csrVal_(0),
      csrRowPtr_(0),
      csrColInd_(0),
      sparseStatus_(CUSPARSE_STATUS_SUCCESS),
      cudaStatus_(cudaSuccess),
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(true)
{ 
    allocateMemory("CudaMatrix::CudaMatrix(CollOfBool)");
    
    kernelSetup s(nnz_ + 1);
    const bool* bool_ptr = thrust::raw_pointer_cast( &bools[0] );
    initBooleanDiagonal<<<s.grid, s.block>>>( csrVal_, csrRowPtr_, csrColInd_,
                bool_ptr, rows_);
    
    createGeneralDescription_("CudaMatrix::CudaMatrix(CollOfBool)");
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
      description_(0),
      operation_(CUSPARSE_OPERATION_NON_TRANSPOSE),
      diagonal_(false)
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
      description_(0),
      operation_(mat.operation_),
      diagonal_(mat.diagonal_)
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
  operation_ = other.operation_;
  diagonal_ = other.diagonal_;

    } // if ( this != &other)
    
    return *this;
}


// Move assignment operator
CudaMatrix& CudaMatrix::operator=(CudaMatrix&& other)
{
    swap(other);
    return *this;
}


// Swap function used for move semantics
void CudaMatrix::swap(CudaMatrix& other) noexcept
{

    std::swap(nnz_, other.nnz_);
    std::swap(csrVal_, other.csrVal_);
    std::swap(csrColInd_, other.csrColInd_);
    std::swap(rows_, other.rows_);
    std::swap(csrRowPtr_, other.csrRowPtr_);
    std::swap(cols_, other.cols_);
    
    operation_ = other.operation_;
    diagonal_ = other.diagonal_;
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

const double* CudaMatrix::csrVal() const {
    return csrVal_;
}
const int* CudaMatrix::csrRowPtr() const {
    return csrRowPtr_;
}
const int* CudaMatrix::csrColInd() const {
    return csrColInd_;
}
double* CudaMatrix::csrVal() {
    return csrVal_;
}
int* CudaMatrix::csrRowPtr() {
    return csrRowPtr_;
}
int* CudaMatrix::csrColInd() {
    return csrColInd_;
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



// TRANSPOSE
CudaMatrix CudaMatrix::transpose() const {
    CudaMatrix out = *this;
    out.operation_ = CUSPARSE_OPERATION_TRANSPOSE;
    return CudaMatrix(std::move(out));
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


// ERROR CHECKING FOR "CudaMatrix * CudaMatrix"
int CudaMatrix::confirmMultSize(const CudaMatrix& lhs, const CudaMatrix& rhs) {
    
    // We need to identify what are the true lhs sizes and rhs sizes wrt "transposity"

    int leftCols = lhs.cols_;
    int leftRows = lhs.rows_;
    int rightCols = rhs.cols_;
    int rightRows = rhs.rows_;
    
    if ( lhs.isTranspose() ) {
  leftCols = lhs.rows_;
  leftRows = lhs.cols_;
    }
    if ( rhs.isTranspose() ) {
  rightCols = rhs.rows_;
  rightRows = rhs.cols_;
    }

    // Check sizes
    if ( leftCols != rightRows ) {
  OPM_THROW(std::runtime_error, "Error in CudaMatrix * CudaMatrix size checking\n" << "\tMatrices of illegal sizes.\n" << "\tlhs.cols_ = " << leftCols << "\n\trhs.rows_ = " << rightRows);
    }

    // If test passed, assign this with correct rows and cols
    this->rows_ = leftRows;
    this->cols_ = rightCols;

    // Return inner size. 
    return leftCols;
}


bool CudaMatrix::isTranspose() const {
    return ( operation_ == CUSPARSE_OPERATION_TRANSPOSE );
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
  return CusparseManager::matrixAddition(lhs,rhs);
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
  return CusparseManager::matrixSubtraction(lhs, rhs);
    }
}


CudaMatrix equelleCUDA::operator*(const CudaMatrix& lhs, const CudaMatrix& rhs)
{
    // If any of them are empty, we return an empty matrix.
    // An empty matrix is interpreted as a correctly sized matrix of zeros.
    // This lets us not worry about empty derivatives for autodiff.
    if ( lhs.isEmpty() || rhs.isEmpty() ) {
        return CudaMatrix();
    }

    // Some functionality is implemented by multiplying with a diagonal matrix
    // from the left. Since csrGemm is a hotspot, we handle these cases more efficient
    // with this function:
    if ( lhs.diagonal_ ) {
        return lhs.diagonalMultiply(rhs);
    }

    return CusparseManager::matrixMultiply(lhs, rhs);
}


// Matrix * vector
CudaArray equelleCUDA::operator*(const CudaMatrix& mat, const CudaArray& vec) {
    //std::cout << "-------MATRIX * VECTOR ---------\n";

     // Check that sizes match - Depend on transpose matrix or not.
    int resultingVectorSize;
    if ( !mat.isTranspose() ) { // NOT transposed
  if ( mat.cols_ != vec.size() ) {
      OPM_THROW(std::runtime_error, "Error in matrix * vector operation as matrix is of size " << mat.rows_ << " by " << mat.cols_ << " and the vector of size " << vec.size());
  }
  resultingVectorSize = mat.rows_;
  //cols = mat.cols_;
    }
    else { // matrix IS transposed
  if ( mat.rows_ != vec.size() ) {
      OPM_THROW(std::runtime_error, "Error in transposed matrix * vector operation as matrix is of size " << mat.cols_ << " by " << mat.rows_ << " and the vector of size " << vec.size());
  }
  resultingVectorSize = mat.cols_;
    }

    
    // Call cusparse matrix-vector operation:
    // y = alpha*op(A)*x + beta*y
    // with alpha=1, beta=0, op=non_transpose
    CudaArray out(resultingVectorSize);
    const double alpha = 1.0;
    const double beta = 0.0;
    mat.sparseStatus_ = cusparseDcsrmv( CUSPARSE,
          mat.operation_,
          mat.rows_, mat.cols_, mat.nnz_, 
          &alpha, mat.description_,
          mat.csrVal_, mat.csrRowPtr_, mat.csrColInd_,
          vec.data(), &beta,
          out.data());
    mat.checkError_("cusparseDcsrmv() in operator*(CudaMatrix, CudaArray)");
    return CudaArray(std::move(out));
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
    return CudaMatrix(std::move(out));
}

CudaMatrix equelleCUDA::operator-(const CudaMatrix& arg) {
    return -1.0*arg;
}



// Diagonal multiplyer:
CudaMatrix CudaMatrix::diagonalMultiply(const CudaMatrix& rhs) const {
    // Make sure we do not call this function if this is not diagonal
    if ( !this->diagonal_ ) {
  OPM_THROW(std::runtime_error, "Error in CudaMatrix::diagonalMultiply\n\tCaller matrix is not diagonal!");
    }

    CudaMatrix out = rhs;
    // this is a square matrix
    kernelSetup s(this->rows_);
    wrapCudaMatrix::diagMult_kernel<<<s.grid, s.block>>>(out.csrVal_,
               out.csrRowPtr_,
               this->csrVal_,
               this->rows_);
    return CudaMatrix(std::move(out));
}

// KERNELS -------------------------------------------------


__global__ void wrapCudaMatrix::initIdentityMatrix(double* csrVal,
               int* csrRowPtr,
               int* csrColInd,
               const int nnz)
{
    const int i = myID();
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
    const int i = myID();
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
    const int i = myID();
    if ( i < rows + 1) {
  csrRowPtr[i] = i;
  if ( i < rows ) {
      csrVal[i] = 1;
      csrColInd[i] = set[i];
  }
    }
}


__global__ void wrapCudaMatrix::initBooleanDiagonal( double* csrVal,
                 int* csrRowPtr,
                 int* csrColInd,
                 const bool* bool_ptr,
                 const int rows) {
    const int i = myID();
    if ( i < rows + 1) {
  csrRowPtr[i] = i;
  if ( i < rows ) {
      csrColInd[i] = i;
      if (bool_ptr[i])
    csrVal[i] = 1;
      else
    csrVal[i] = 0;
  }
    }
}


__global__ void wrapCudaMatrix::diagMult_kernel( double* csrVal,
             const int* csrRowPtr,
             const double* diagVals,
             const int total_rows) 
{
    const int row = myID();
    if ( row < total_rows ) {
  for (int i = csrRowPtr[row]; i < csrRowPtr[row+1]; i++) {
      csrVal[i] = diagVals[row] * csrVal[i];
  }
    }
}

