
#ifndef EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED

//#include <thrust/device_ptr.h>
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <string>
#include <fstream>
#include <iterator>


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
private:
    //thrust::host_vector<double> host_vec;
    //thrust::device_vector<double> dev_vec;
    double* values;
    int size;
};

inline CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out(lhs.getSize());
    for(int i = 0; i < out.getSize(); ++i) {
	out.setValue(i, lhs.getValue(i) - rhs.getValue(i));
    }
    return out;
}


#endif // EQUELLERUNTIMECUDA_CUDA_HEADER_INCLUDED
