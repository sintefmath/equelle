
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
    ~CollOfScalar();

    void setValuesFromFile(std::istream_iterator<double> begin, 
			   std::istream_iterator<double> end);
    void setValuesUniform(double val, int size);
    double getValue(int index) const;
    void setValue(int index, double value);
    int getSize() const;
    double* getDevValues() const;
    //double* getHostValues() const;
    void copyToHost() const ;
    thrust::device_vector<double> dev_vec;
private:
    //thrust::host_vector<double> host_vec;
    //thrust::device_vector<double> dev_vec;
    double* values;
    int size;
    double* dev_values;

};

inline CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar *out = new CollOfScalar(lhs.getSize());
    //for(int i = 0; i < out->getSize(); ++i) {
    //    out->setValue(i, lhs.getValue(i) - rhs.getValue(i));
    //}
    
    double* lhs_dev = lhs.getDevValues();
    double* rhs_dev = rhs.getDevValues();
    double* out_dev = out->getDevValues();
    
    cublasHandle_t blasHandle;
    cublasStatus_t blasStatus;

    // creating necessary stuff - cuBLAS:
    blasStatus = cublasCreate( &blasHandle );
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error creating cublas!\n");
	exit(0);
    }

    // out = lhs
    blasStatus = cublasDcopy( blasHandle, out->getSize() , lhs_dev, 1, out_dev, 1);
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error: operator- : out_dev = lhs_dev\n");
	exit(0);
    }

    // x = alpha * p + x
    // out = -1 * rhs + lhs
    double negOne = -1.0;
    blasStatus = cublasDaxpy( blasHandle, out->getSize(), &negOne, rhs_dev, 1, out_dev, 1);
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("CG - Error x = x + alpha*p\n");
	return false;
    }

    blasStatus = cublasDestroy( blasHandle );
    if ( blasStatus != CUBLAS_STATUS_SUCCESS ) {
	printf("Error destroying cublas!\n");
	exit(0);
    }

    return *out;
}

namespace
havahol_helper {


}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
