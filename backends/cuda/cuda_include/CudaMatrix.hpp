
#ifndef EQUELLE_CUDAMATRIX_HEADER_INCLUDED
#define EQUELLE_CUDAMATRIX_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

// This file include a global variable for the cusparse handle
// It has to be created before any CudaMatrix objects are decleared!
#include "equelleTypedefs"

#include <cusparse_v2.h>


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
		    int rows, int cols, int nnz);

	//! Copy assignment operator
	CudaMatrix& operator= (const CudaMatrix& other);
	


	//! Destructor
	~CudaMatrix();

	friend CudaMatrix operator+ (const CudaMatrix& lhs, const CudaMatrix& rhs);
	friend CudaMatrix operator* (const CudaMatrix& lhs, const CudaMatrix& rhs);
    private:
	int rows_;
	int cols_;
	int nnz_;

	double* csrVal_;
	int* csrRowPtr_;
	int* csrColInd_;
	
	cusparseMatDescr_t description_;
	
	// Error handling:
	mutable cusparseStatus_t sparseStatus_;
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;
	
    }



} // namespace equelleCUDA


#endif // EQUELLE_CUDAMATRIX_HEADER_INCLUDED
