
#ifndef EQUELLE_TYPEDEFS_HEADER_INCLUDED
#define EQUELLE_TYPEDEFS_HEADER_INCLUDED

#include <cuda.h>

#include <thrust/device_vector.h>
#include <vector>

namespace equelleCUDA
{
    /*!
      Define max number of threads in a kernel block:
    */
    const int MAX_THREADS = 512;
    //const int MAX_THREADS = 7;
    
    
    typedef thrust::device_vector<bool> CollOfBool;
    
    typedef double Scalar;
    typedef bool Bool;
    typedef std::string String;
    
    
    
    struct kernelSetup 
    {
	const dim3 block;
	const dim3 grid;
	
	kernelSetup(int threads_needed) 
	: block(equelleCUDA::MAX_THREADS),
	  grid( (int)(( threads_needed + equelleCUDA::MAX_THREADS - 1)/equelleCUDA::MAX_THREADS) )
	{}
    };
    
} // namespace equelleCUDA


#endif // EQUELLE_TYPEDEFS_HEADER_INCLUDED
