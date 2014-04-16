
#ifndef EQUELLE_CUDAMATRIX_HEADER_INCLUDED
#define EQUELLE_CUDAMATRIX_HEADER_INCLUDED

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

// This file include a global variable for the cusparse handle
// It has to be created before any CudaMatrix objects are decleared!
#include "equelleTypedefs.hpp"


namespace equelleCUDA {

    //! Struct for holding transferring CudaMatrix to host.
    /*!
      This struct is only for conveniency for testing purposes.
    */
    struct hostMat
    {
	//! Values
	std::vector<double> vals;
	//! Row pointers
	std::vector<int> rowPtr;
	//! Column indices
	std::vector<int> colInd;
	//! Number of nonzeros 
	int nnz;
	//! Number of rows
	int rows;
	//! Number of columns
	int cols;
    };
} // namespace equelleCUDA
    


namespace equelleCUDA {

    //! Class for storing a Matrix on the device
    /*!
      This class stores a rows*cols sized Matrix in CSR (Compressed Sparse Row) 
      format consisting of nnz nonzero values. 
      It consists of three data arrays living on the device:\n
      - csrVal: The data array which holds the nonzero values of the matrix in 
      row major format. Size: nnz.\n
      - csrRowPtr: Holds one index per row saying where in csrVal each row starts.
      Row i will therefore start on scrVal[csrRowPtr[i]] and go up to (but not
      including scrVal[csrRowPtr[i+1]]. Size: rows + 1.\n
      - csrColInd: For each value in csrVal, csrColInd holds the column index. Size: nnz.  
    */
    class CudaMatrix
    {
    public:
	//! Default constructor
	/*!
	  This constructor do not allocate memory on the device, and pointers for the 
	  matrix itself is initialized to zero.
	*/
	CudaMatrix();

	//! Copy constructor
	/*!
	  Allocate memory on the device and copy the data from the input matrix, as long
	  as the input arrays are not zero.
	*/
	CudaMatrix(const CudaMatrix& mat);
	
	//! Constructor for testing using host arrays
	/*!
	  This constructor takes pointers to host memory as input, allocates the same 
	  amount of memory on the device and copy the data. 

	  This constructor is mostly used for testing.
	*/
	CudaMatrix( const double* val, const int* rowPtr, const int* colInd,
		    const int nnz, const int rows, const int cols);

	//! Constructor creating a identity matrix of the given size.
	/*!
	  Allocates and create a size by size identity matrix.

	  This is the constructor that should be used in order to create a primary
	  variable for Automatic Differentiation.
	*/
	CudaMatrix( const int size);

	//! Copy assignment operator
	/*!
	  Allocates and copies the device memory from the input matrix to this. 
	  Does also perform checks for matching array sizes and self assignment.
	*/
	CudaMatrix& operator= (const CudaMatrix& other);
	
       
	//! Destructor
	/*!
	  Free all device memory allocated by the given object.
	*/
	~CudaMatrix();


	//! The number of non-zero entries in the matrix.
	int nnz() const;
	//! The number of rows in the matrix.
	int rows() const;
	//! The number of columns in the matrix.
	int cols() const;

	//! Copies the device memory to host memory in a hostMat struct.
	hostMat toHost() const;

	friend CudaMatrix operator+(const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator-(const CudaMatrix& lhs, const CudaMatrix& rhs);
	// C = A + beta*B
	friend CudaMatrix cudaMatrixSum(const CudaMatrix& lhs,
					const CudaMatrix& rhs,
					const double beta);
	
	friend CudaMatrix operator*(const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator*(const CudaMatrix& lhs, const Scalar rhs);
	friend CudaMatrix operator*(const Scalar lhs, const CudaMatrix& rhs);
    private:
	int rows_;
	int cols_;
	int nnz_;

	double* csrVal_;
	int* csrRowPtr_;
	int* csrColInd_;
	
	// Error handling:
	mutable cusparseStatus_t sparseStatus_;
	mutable cudaError_t cudaStatus_;

	cusparseMatDescr_t description_;

	void checkError_(const std::string& msg) const;
	void createGeneralDescription_(const std::string& msg);
	
    }; // class CudaMatrix
    
    
    //! Matrix + Matrix operator
    /*!
      Makes a call to cudaMatrixSum with beta = 1.0.
      \sa cudaMatrixSum
    */
    CudaMatrix operator+(const CudaMatrix& lhs, const CudaMatrix& rhs);
    //! Matrix - Matrix operator
    /*!
      Makes a call to cudaMatrixSum with beta = -1.0.
      \sa cudaMatrixSum
     */
    CudaMatrix operator-(const CudaMatrix& lhs, const CudaMatrix& rhs);
    //! Summation of two sparse matrices.
    /*!
      Performs a sparse matrix + sparse matrix operation of the form 
      \code lhs + beta*rhs \endcode
      where beta is a constant. We use beta = 1.0 for addition and beta = -1.0 for
      subtraction.

      In order to add the two matrices we do a two step use of the cusparse library.\n
      1) Find the number of non-zeros for each row of the resulting matrix.\n
      2) Allocate memory for result and add matrices.
    */
    CudaMatrix cudaMatrixSum( const CudaMatrix& lhs,
			      const CudaMatrix& rhs,
			      const double beta);
    //! Matrix * Matrix operator
    /*!
      Performs a sparse matrix * sparse matrix operation in two steps by using the
      cusparse library.\n
      1) Find the number of non-zeros for each row of the resulting matrix.\n
      2) Allocate memory for the result and find the matrix product.
    */
    CudaMatrix operator*(const CudaMatrix& lhs, const CudaMatrix& rhs);

    //! Multiplication with Matrix and scalar
    /*!
      Make a call to the kernel multiplying a scalar to a cudaArray, since all non-zero
      entries are at the same positions, and the values are continuously stored in memory.
      \sa wrapCudaArray::scalMultColl_kernel
     */
    CudaMatrix operator*(const CudaMatrix& lhs, const Scalar rhs);
    //! Multiplication with scalar and Matrix
    /*!
      Make a call to the kernel multiplying a scalar to a cudaArray, since all non-zero
      entries are at the same positions, and the values are continuously stored in memory.
    */
    CudaMatrix operator*(const Scalar lhs, const CudaMatrix& rhs);

    //! Kernels and other related functions to for CudaMatrix:
    namespace wrapCudaMatrix
    {

	//! Kernel for initializing an identity matrix.
	/*!
	  For allocated memory for a nnz times nnz matrix, this kernel fills in the 
	  correct values in the three input arrays.
	  \param[out] csrVal Matrix values. Here: nnz copies of 1.0.
	  \param[out] csrRowPtr Indices for where in the csrVal array each row starts.
	  Here: csrRowPtr[i] = i for i=0..nnz
	  \param[out] csrColInd The column index for each entry in csrVal.
	  Here: csrColInd[i] = i
	  \param[in] nnz Number of nonzero entries. Same as the number of rows and
	  columns as well.
	*/
	__global__ void initIdentityMatrix(double* csrVal,
					   int* csrRowPtr,
					   int* csrColInd,
					   const int nnz);

    } // namespace wrapCudaMatrix

} // namespace equelleCUDA


#endif // EQUELLE_CUDAMATRIX_HEADER_INCLUDED
