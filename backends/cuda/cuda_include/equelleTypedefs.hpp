
#ifndef EQUELLE_TYPEDEFS_HEADER_INCLUDED
#define EQUELLE_TYPEDEFS_HEADER_INCLUDED

#include <cuda.h>
#include <cusparse_v2.h>

#include <thrust/device_vector.h>
#include <vector>


//! Namespace for the entire CUDA Back-End for Equelle.
namespace equelleCUDA
{
    /*!
      Define max number of threads in a kernel block:
    */
    const int MAX_THREADS = 512;
    //const int MAX_THREADS = 7;

    // This global variable is decleared in src/wrapEquelleRuntime.cu
    //! Handle needed for cusparse library calls.
    extern cusparseHandle_t CUSPARSE;
    
    //! Use the Equelle name Scalar for double.
    typedef double Scalar;
    //! Use the Equelle name Bool for bool.
    typedef bool Bool;
    //! Add capitalized String as it is used in serial back-end.
    typedef std::string String;
    
   
    //! Collection of booleans as a thrust::device_vector<bool>
    typedef thrust::device_vector<Bool> CollOfBool;
    

    //! Grid and block sizes for kernels.
    /*!
      This struct is used to calculate the grid and block sizes we need in order 
      to launch a kernel.

      Its constructor takes the needed number of threads as input, and it uses the 
      MAX_THREADS constant to set block sizes, and ensures enough blocks are launched.
      
      \sa MAX_THREADS
    */
    struct kernelSetup 
    {
	//! Block size
	const dim3 block;
	//! Grid size
	const dim3 grid;
	
	//! Constructor for struct kernelSetup.
	kernelSetup(int threads_needed) 
	: block(equelleCUDA::MAX_THREADS),
	  grid( (int)(( threads_needed + equelleCUDA::MAX_THREADS - 1)/equelleCUDA::MAX_THREADS) )
	{}
    };

    //! Enumerator for specification of reduction operation.
    enum EquelleReduce { SUM, PRODUCT, MAX, MIN };

    
    
} // namespace equelleCUDA


#endif // EQUELLE_TYPEDEFS_HEADER_INCLUDED
