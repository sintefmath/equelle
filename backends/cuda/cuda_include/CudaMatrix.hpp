
#ifndef EQUELLE_CUDAMATRIX_HEADER_INCLUDED
#define EQUELLE_CUDAMATRIX_HEADER_INCLUDED

#include <vector>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>

#include <Eigen/Sparse>

// This file include a global variable for the cusparse handle
// It has to be created before any CudaMatrix objects are decleared!
#include "equelleTypedefs.hpp"
//#include "CollOfScalar.hpp" // forward declaration of this class.
#include "CudaArray.hpp"

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
    
    // Forward declaration of CollOfScalar:
    class CollOfScalar;
    

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
	explicit CudaMatrix( const double* val, const int* rowPtr, const int* colInd,
			     const int nnz, const int rows, const int cols);

	//! Constructor from Eigen sparse matrix
 	/*!
	  Creates a CudaMatrix from an Eigen Sparse Matrix. Eigen use column
	  major formats by defaults, and we therefore have to copy the matrix 
	  over to row major format before copying it to the device.
	  
	  This constructor is needed for the DeviceHelperOps matrices, since the
	  Opm::HelperOps builds matrices of type Eigen::SparseMatrix<Scalar>. The
	  transformation to row-major could also have been done on the GPU, but that
	  would require more code, as well as this is an operation that happens only
	  in the constructor of the EquelleRuntimeCUDA class.
	*/
	explicit CudaMatrix( const Eigen::SparseMatrix<Scalar>& eigen);
	

	//! Constructor creating a identity matrix of the given size.
	/*!
	  Allocates and create a size by size identity matrix.

	  This is the constructor that should be used in order to create a primary
	  variable for Automatic Differentiation.
	*/
	explicit CudaMatrix( const int size);

  //! Constructs a matrix of size rows*cols with nnz non-zero elements.
  /*!
    The constructor allocates memory, but does not initialize it.
    Perhaps it is better to pass pointers for the column index, row pointer
    and csr values as well.
  */
  explicit CudaMatrix( const int rows, const int cols, const int nnz);


	//! Constructor creating a diagonal matrix from a CollOfScalar
	/*!
	  Allocates memory for a diagonal matrix, and insert the CollOfScalar 
	  values on the diagonal elements. This is regardless if the CollOfScalar has 
	  a derivative or not, we only use its values.
	*/
	explicit CudaMatrix( const CollOfScalar& coll ); 

	//! Constructor for creating a diagonal matrix from a CudaArray
	/*!
	  Allocates memory for a diagonal matrix and insert the CudaArray values on
	  the diagonal elements.
	*/
	explicit CudaMatrix( const CudaArray& array);

	//! Construct diagonal matrix from CollOfBools
	/*!
	  The diagonal entries are 1 for where array is true and 0 where array is false.
	*/
	explicit CudaMatrix( const CollOfBool& array);

	// Constructor for making a restriction matrix.
	/*!
	  Operations such as On and Extend can be done by applying a restriction or 
	  prolongation operator. This constructor creates a restriction operator based
	  on a vector of indices of which rows should be in the restricted matrix. The
	  size input should be the number of rows in the origianal system.
	  
	  A prolongation matrix can be created by creating a restriction matrix for
	  the opposite calculation (restriction for result to original), and then
	  using the transpose matrix instead.
	*/
	explicit CudaMatrix( const thrust::device_vector<int> set,
			     const int full_size);

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

	//! Const pointer to the matrix data on the device
	const double* csrVal() const ;
	//! Const pointer to the row pointers on the device
	const int* csrRowPtr() const ;
	//! Const pointer to the column indices on the device
	const int* csrColInd() const ;
	//! Pointer to the matrix data on the device
	double* csrVal();
	//! Pointer to the row pointer on the device
	int* csrRowPtr();
	//! Pointers to the column indices on the device
	int* csrColInd();
	
	//! Check if the matrix holds values or not
	bool isEmpty() const;

	//! Copies the device memory to host memory in a hostMat struct.
	hostMat toHost() const;
	
	//! Returns the transpose of the matrix.
	/*!
	  Uses the cusparse routine to convert between row-major and column-major
	  formats, which is the same as finding the row major format of the transpose
	  of a row-major matrix.
	*/
	CudaMatrix transpose() const;

	friend CudaMatrix operator+(const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator-(const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator*(const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaArray operator*(const CudaMatrix& mat, const CudaArray& vec);
	friend CudaMatrix operator*(const CudaMatrix& lhs, const Scalar rhs);
	friend CudaMatrix operator*(const Scalar lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator-(const CudaMatrix& arg);
	friend class CusparseManager;

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
	cusparseOperation_t operation_;
	bool diagonal_;

	void checkError_(const std::string& msg) const;
	void checkError_(const std::string& msg, const std::string& caller) const;
	void createGeneralDescription_(const std::string& msg);

	void allocateMemory(const std::string& caller);
	
	// Check that lhs*rhs is legal, assign this with correct rows_ and cols_. 
	// It returns the inner size of the matrix multiplication.
	int confirmMultSize(const CudaMatrix& lhs, const CudaMatrix& rhs);
	// Check if the matrix is a transpose or not.
	bool isTranspose() const;

	CudaMatrix diagonalMultiply(const CudaMatrix& rhs) const;
	
    }; // class CudaMatrix
    
    
    //! Matrix + Matrix operator
    /*!
      Matrices are allowed to be empty, and will then be interpreted as 
      correctly sized matrices filled with zeros.
    */
    CudaMatrix operator+(const CudaMatrix& lhs, const CudaMatrix& rhs);

    //! Matrix-  Matrix operator
    /*!
      Matrices are allowed to be empty, and will then be interpreted as
      correctly sized matrices filled with zeros.
     */
    CudaMatrix operator-(const CudaMatrix& lhs, const CudaMatrix& rhs);

    //! Matrix * Matrix operator
    /*!
      Performs a sparse matrix * sparse matrix operation in two steps by using the
      cusparse library.\n
      1) Find the number of non-zeros for each row of the resulting matrix.\n
      2) Allocate memory for the result and find the matrix product.

      The matrices are allowed to be empty, and an empty matrix is
      interpreted as a correctly sized matrix of zeros.
      This lets us not worry about empty derivatives for autodiff.
    */
    CudaMatrix operator*(const CudaMatrix& lhs, const CudaMatrix& rhs);

    //! Matrix * vector operator
    /*!
      Performs a matrix vector product, where the vector is represented by a 
      CudaArray object.

      It is useful for some of the neighbour relations such as gradient and
      divergence.
    */
    CudaArray operator*(const CudaMatrix& mat, const CudaArray& vec);

    //! Multiplication with Matrix and scalar
    /*!
      Make a call to the kernel multiplying a scalar to a cudaArray, since all non-zero
      entries are at the same positions, and the values are continuously stored in memory.

      The matrix is not allowed to be empty.
      \sa wrapCudaArray::scalMultColl_kernel
     */
    CudaMatrix operator*(const CudaMatrix& lhs, const Scalar rhs);
    //! Multiplication with scalar and Matrix
    /*!
      Make a call to the kernel multiplying a scalar to a cudaArray, since all non-zero
      entries are at the same positions, and the values are continuously stored in memory.

      The matrix is not allowed to be empty.
      \sa wrapCudaArray::scalMultColl_kernel
    */
    CudaMatrix operator*(const Scalar lhs, const CudaMatrix& rhs);
    
    //! Unary minus
    /*!
      Returns -1.0* arg;
    */
    CudaMatrix operator-(const CudaMatrix& arg);



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
	  Here: csrColInd[i] = i for i=0..nnz
	  \param[in] nnz Number of nonzero entries. Same as the number of rows and
	  columns as well.
	*/
	__global__ void initIdentityMatrix( double* csrVal,
					    int* csrRowPtr,
					    int* csrColInd,
					    const int nnz);

	//! Kernel for initializing a diagonal matrix from a CollOfScalar value array
	/*!

	  \param[out] csrVal Matrix values: Here copy of scalars:
	  csrVal[i] = scalars[i] for i=0..nnz-1
	  \param[out] csrRowPtr Indices for where in the csrVal array each row starts.
	  Here: csrRowPtr[i] = i for i=0..nnz
	  \param[out] csrColInd The column index for each entry in csrVal.
	  Here: csrColInd[i] = i for i=0..nnz-1
	  \param[in] scalars The input collection of scalar that will be the diagonal
	  entries.
	  \param[in] nnz Number of nonzero entries. Same as the number of rows and
	  columns as well.
	*/
	__global__ void initDiagonalMatrix( double* csrVal,
					    int* csrRowPtr,
					    int* csrColInd,
					    const double* scalars,
					    const int nnz);


	//! Restriction matrix initialization kernel
	/* Initialize a restriction matrix based on a set of indices.
	   A restriction matrix is such that if multiplied from the left to another
	   matrix, the resulting matrix will be a subset of the original matrix's rows.
	   	   
	   A restriction matrix will therefore have less rows than columns, and it
	   will have one and only one value in each row. This value is one and it 
	   is placed in the column corresponding to the row index we want to have 
	   in the resulting matrix at the current row.
	   
	   \param[out] csrVal rows copies of 1.0
	   \param[out] csrRowPtr where in csrVal each row starts. Since each row has
	   only one value, this will therefore be an array [0,1,2,...,rows]
	   \param[out] csrColInd Column index for each entry in the matrix, and is 
	   a copy of the parameter set.
	   \param[in] set The indices that the restriction matrix should map to.
	   \param[in] rows Number of rows in the restriction matrix, and also the
	   size of parameter set.
	*/
	__global__ void initRestrictionMatrix( double* csrVal,
					       int* csrRowPtr,
					       int* csrColInd,
					       const int* set,
					       const int rows );

	//! Kernel for initializing diagonal matrix from booleans
	/*
	  Use the array of booleans to insert ones or zeros at the diagonal elements.
	  
	  \param[out] csrVal Diagonal elements og 1.0 or 0.0 and size rows.
	  \param[out] csrRowPtr Index in csrVal and csrColInd that is the start of
	  each row. Here: csrRowPtr[i] = i for i = 0,1,...,rows.
	  \param[out] csrColInd Column index of each element in csrVal.
	  Here: csrColInd[i] = i for i = 0,1,...,rows-1
	  \param[in] bool_ptr The array of boolean values 
	  \param[in] rows Number of rows in the matrix.
	*/
	__global__ void initBooleanDiagonal( double* csrVal,
					     int* csrRowPtr,
					     int* csrColInd,
					     const bool* bool_ptr,
					     const int rows);
	
	//! Kernel for computing diagonal matrix-matrix multiplication
	/*
	  Multiplication with a diagonal matrix from the right preserve non-zero 
	  patterns, and we therefore treat this operation as a special case.
	  We therefore make a copy of the rhs matrix to the result prior to
	  calling this kernel.
	  The operation corresponds to multiply each element in the diagonal matrix
	  to every element in the other matrix's corresponding row.
	  
	  \param[in,out] csrVal Right-hand side matrix non-zero values,
	  will be overwritten by the result.
	  \param[in] csrRowPtr Points to where in csrVal each row starts.
	  \param[in] diagVals Diagonal values from lhs diagonal matrix.
	  \param[in] total_row Total number of rows in resulting matrix.
	*/
	__global__ void diagMult_kernel( double* csrVal,
					 const int* csrRowPtr,
					 const double* diagVals,
					 const int total_rows);



    } // namespace wrapCudaMatrix

} // namespace equelleCUDA


#endif // EQUELLE_CUDAMATRIX_HEADER_INCLUDED
