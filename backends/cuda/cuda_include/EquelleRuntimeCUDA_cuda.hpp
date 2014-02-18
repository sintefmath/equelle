
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
__global__ void minus_kernel(double* out, double* rhs, int size);
__global__ void plus_kernel(double* out, double* rhs, int size);
__global__ void multiplication_kernel(double* out, double* rhs, int size);
__global__ void division_kernel(double* out, double* rhs, int size);



class CollOfScalar
{
public:
    CollOfScalar();
    CollOfScalar(int size);
    CollOfScalar(const CollOfScalar& coll);
    ~CollOfScalar();

    void setValuesFromFile(std::istream_iterator<double> begin, 
			   std::istream_iterator<double> end);
    void setValuesUniform(double val);

    int size() const;
    double* data() const;
    std::vector<double> copyToHost() const ;

    int grid() const;
    int block() const;
private:
    int size_;
    double* dev_values;

    // Error handling
    mutable cudaError_t cudaStatus;
    void checkError(const std::string& msg) const;

    // Use 1D kernel grids for arithmetic operations
    int grid_x_;
    int block_x_;

};

// Operation overloading
CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs);
CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs);


namespace havahol_helper 
{
    
    // Define max number of threads in a kernel block:
    const int MAX_THREADS = 512;

}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
