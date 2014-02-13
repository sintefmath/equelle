
#ifndef EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED

//#include <thrust/device_ptr.h>
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublas_v2.h>
#include <cuda.h>

#include <string>
#include <fstream>
#include <iterator>

// This is the header file for cuda!

//#include "EquelleRuntimeCUDA.hpp"


// Kernel declarations:
__global__ void minus_kernel(double* out, double* rhs, int size);

class CollOfScalar
{
public:
    // CollOfScalar();
    CollOfScalar(int size);
    CollOfScalar(const CollOfScalar& coll);
    ~CollOfScalar();

    void setValuesFromFile(std::istream_iterator<double> begin, 
			   std::istream_iterator<double> end);
    void setValuesUniform(double val);
    int getSize() const;
    double* getDevValues() const;
    void copyToHost(double* values) const ;

    int grid();
    int block();
private:
    int size;
    double* dev_values;

    // Use 1D kernel grids for arithmetic operations
    int grid_x;
    int block_x;

};

CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs);
/*CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out = lhs;
    //double* lhs_dev = lhs.getDevValues();
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();

    dim3 block(out.block());
    dim3 grid(out.grid());
    std::cout << "Calling minus_kernel!\n";
    minus_kernel <<<grid, block>>>(out_dev, rhs_dev, out.getSize());
    
}/**/

/*inline CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    
    //CollOfScalar out(lhs.getSize());
    CollOfScalar out = lhs;
    
    std::cout << "MINUS!\n";

    //double* lhs_dev = lhs.getDevValues();
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out.getDevValues();
    
    cublasHandle_t blasHandle;
    cublasStatus_t blasStatus;

    // creating necessary stuff - cuBLAS:
    blasStatus = cublasCreate( &blasHandle );
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error creating cublas!\n");
	exit(0);
    }

    // out = lhs
    //blasStatus = cublasDcopy( blasHandle, out.getSize() , lhs_dev, 1, out_dev, 1);
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error: operator- : out_dev = lhs_dev\n");
	exit(0);
    }

    // x = alpha * p + x
    // out = -1 * rhs + lhs
    double negOne = -1.0;
    blasStatus = cublasDaxpy( blasHandle, out.getSize(), &negOne, rhs_dev, 1, out_dev, 1);
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("CG - Error x = x + alpha*p\n");
	return false;
    }

    blasStatus = cublasDestroy( blasHandle );
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error destroying cublas!\n");
	exit(0);
    }

    return out;
}/* comment stop! */



namespace havahol_helper 
{
    
    // Define max number of threads in a kernel block:
    const int MAX_THREADS = 512;

}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
