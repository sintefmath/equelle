
#ifndef EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED

//#include <thrust/device_ptr.h>
//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>

#include <cublas_v2.h>
#include <cuda.h>

#include <string>
#include <fstream>
#include <iterator>

// This is the header file for cuda!

//#include "EquelleRuntimeCUDA.hpp"


// Kernel declarations:
//! CUDA kernel for the minus operator
namespace equelleCUDA {
    __global__ void minus_kernel(double* out, const double* rhs, const int size);
    __global__ void plus_kernel(double* out, const double* rhs, const int size);
    __global__ void multiplication_kernel(double* out, const double* rhs, const int size);
    __global__ void division_kernel(double* out, const double* rhs, const int size);
}


//! CollOfScalar class for the Equelle CUDA Back-end
/*!
  Class for storing and handeling CollectionOfScalar variables in Equelle.
  The class is part of the CUDA back-end of the Equelle compiler.
 */


class CollOfScalar
{
public:
    //! Default constructor
    CollOfScalar();
    explicit CollOfScalar(const int size);
    
    explicit CollOfScalar(const int size, const int value);

    //! Constructor from std::vector
    /*! Used for initialize CollOfScalar when using unit tests. */
    explicit CollOfScalar(const std::vector<double>& host_vec);
    CollOfScalar(const CollOfScalar& coll);  
    ~CollOfScalar();

    int size() const;
    const double* data() const;
    double* data();
    std::vector<double> copyToHost() const;

    //! Returns the CUDA gridDim.x size for kernel calls involving this collection
    int grid() const;
    //! Returns the CUDA blockDim.x size for kernel calls involving this collection
    int block() const;
private:
    int size_;
    double* dev_values_;

    // Use 1D kernel grids for arithmetic operations
    int block_x_;
    int grid_x_;

    // Error handling
    mutable cudaError_t cudaStatus_;
    void checkError_(const std::string& msg) const;


};


// Operation overloading
CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs);


namespace equelleCUDA
{
    
    // Define max number of threads in a kernel block:
    const int MAX_THREADS = 512;

}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
