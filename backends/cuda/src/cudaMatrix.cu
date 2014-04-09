#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include <opm/core/utility/ErrorMacros.hpp>

#include <vector>
#include <iostream>
#include <string>

#include "CudaMatrix.hpp"


using namespace equelleCUDA;

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


// Destructor
CudaMatrix::~CudaMatrix() {
    

}

int CudaMatrix::getNnz() const {
    return nnz_;
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