
#ifndef EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED

//#include <thrust/device_ptr.h>
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cublas_v2.h>

#include <string>
#include <fstream>
#include <iterator>

// This is the header file for cuda!

//#include "EquelleRuntimeCUDA.hpp"

class CollOfScalar
{
public:
    CollOfScalar();
    CollOfScalar(int size);
    CollOfScalar(const CollOfScalar& coll);
    ~CollOfScalar();

    void setValuesFromFile(std::istream_iterator<double> begin, 
			   std::istream_iterator<double> end);
    void setValuesUniform(double val, int size);
    double getValue(int index) const;
    void setValue(int index, double value);
    int getSize() const;
    double* getRawPtr();
    //double* getRawPtr() const;
    double* getHostValues() ;
    void copyToHost() const ;
    //thrust::device_vector<double> dev_vec;
    void wrapPtrIntoVec(double* out_dev);

private:
    //thrust::host_vector<double> host_vec;
    thrust::device_vector<double> dev_vec;
    double* values;
    int size;
    double* dev_values;

};

inline CollOfScalar operator-(CollOfScalar lhs, CollOfScalar rhs) {
    //CollOfScalar out = new CollOfScalar(lhs.getSize());
    CollOfScalar out(lhs.getSize());
    //for(int i = 0; i < out->getSize(); ++i) {
    //    out->setValue(i, lhs.getValue(i) - rhs.getValue(i));
    //}
    
    std::cout << "MINUS! with sizes: " << out.getSize() << " = " << lhs.getSize() << " - ";
    std::cout << rhs.getSize() << "\n";

    double* lhs_dev = lhs.getRawPtr();
    double* rhs_dev = rhs.getRawPtr();
    double* out_dev = out.getRawPtr();
    
    cublasHandle_t blasHandle;
    cublasStatus_t blasStatus;

    // creating necessary stuff - cuBLAS:
    blasStatus = cublasCreate( &blasHandle );
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error creating cublas!\n");
	exit(0);
    }

    // out = lhs
    blasStatus = cublasDcopy( blasHandle, out.getSize() , lhs_dev, 1, out_dev, 1);
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
    
    //CollOfScalar outX = out;
    // Make sure that out.dev_vec is wrapped around the raw buffer!
    std::cout << "End of minus: out.getSize() = " << out.getSize() << "\n";

    double* a_dev = out.getRawPtr();
    double* a = (double*)malloc(out.getSize()*sizeof(double));
    cudaError_t stat = cudaMemcpy(a, a_dev, sizeof(double)*out.getSize(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < out.getSize(); i++) {
	std::cout << a[i] << " ";
    }
    std::cout << std::endl;
    out.wrapPtrIntoVec(out_dev);

    cudaPointerAttributes attr;
    stat = cudaPointerGetAttributes(&attr, a_dev);
    if (stat != cudaSuccess) {
	std::cout << "Error in getAttribute\n\t";
	std::cout << "Error code: " << cudaGetErrorString(stat) << std::endl;
    }
    std::cout << "a_dev lives on: " << attr.memoryType << std::endl;

    return out;
}

namespace
havahol_helper {


}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
