
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
    struct hostMat
    {
	std::vector<double> vals;
	std::vector<int> rowPtr;
	std::vector<int> colInd;
	int nnz;
	int rows;
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
	CudaMatrix();

	//! Copy constructor
	CudaMatrix(const CudaMatrix& mat);
	
	//! Constructor for testing using host arrays
	CudaMatrix( const double* val, const int* rowPtr, const int* colInd,
		    const int nnz, const int rows, const int cols);

	//! Copy assignment operator
	CudaMatrix& operator= (const CudaMatrix& other);
	
	

	//! Destructor
	~CudaMatrix();

	int nnz() const;
	int rows() const;
	int cols() const;

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
	void createGeneralDescription(const std::string& msg);
	
    }; // class CudaMatrix
    
    CudaMatrix operator+(const CudaMatrix& lhs, const CudaMatrix& rhs);
    CudaMatrix operator-(const CudaMatrix& lhs, const CudaMatrix& rhs);
    CudaMatrix cudaMatrixSum( const CudaMatrix& lhs,
			      const CudaMatrix& rhs,
			      const double beta);
    CudaMatrix operator*(const CudaMatrix& lhs, const CudaMatrix& rhs);
    CudaMatrix operator*(const CudaMatrix& lhs, const Scalar rhs);
    CudaMatrix operator*(const Scalar lhs, const CudaMatrix& rhs);

} // namespace equelleCUDA


#endif // EQUELLE_CUDAMATRIX_HEADER_INCLUDED
