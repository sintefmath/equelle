
#ifndef EQUELLE_TYPEDEFS_HEADER_INCLUDED
#define EQUELLE_TYPEDEFS_HEADER_INCLUDED


#include <thrust/device_vector.h>
#include <vector>

typedef thrust::device_vector<bool> CollOfBool;

/*class CollOfBool : public thrust::device_vector<bool>
{
public:
    CollOfBool(thrust::device_vector<bool>::iterator begin,
	       thrust::device_vector<bool>::iterator end) 
	: thrust::device_vector<bool>(begin, end)
    {};
    
    std::vector<bool> stdToHost();
    };*/

typedef double Scalar;
typedef bool Bool;
typedef std::string String;


#endif // EQUELLE_TYPEDEFS_HEADER_INCLUDED
