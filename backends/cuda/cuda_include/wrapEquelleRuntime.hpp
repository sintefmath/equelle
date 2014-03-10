#ifndef EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED
#define EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"


namespace equelleCUDA
{
    //! Wrapper for calling the trinaryIf kernel
    /*!
      Set up the kernel for a trinaryIf. Assumes that the given input parameters 
      satisfies the criterias as being the same size and not containing any illigal 
      values.

      \param predicate A Collection of Booleans often written as a test.
      \param iftrue Values the output elements should take if the test is true
      \param iffalse Values the output elements should take if the test is false.
      return A collection of Scalars with elements form iftrue and iffalse according
      to predicate.
    */
    CollOfScalar trinaryIfWrapper( const CollOfBool& predicate,
				   const CollOfScalar& iftrue,
				   const CollOfScalar& iffalse);

    //! Kernel for trinaryIf
    /*!
      Kernel for evaluating a trinaryIf statement. The output value out takes the 
      value from iftrue or iffalse depending on the boolean in predicate. For each 
      valid index i we have
      \code
      if (predicate[i])
          out[i] = iftrue[i]
      else
          out[i] = iffalse[i]
      \endcode
      
      \param[out] out Result values
      \param[in] predicate Booleans to indicate which values to assign to out.
      \param[in] iftrue Values to be assigned to out if predicate is true
      \param[in] iffalse Values to be assigned to out if predicate is false
      \param[in] size Size of the above arrays.
    */
    __global__ void trinaryIfKernel( double* out,
				     const bool* predicate,
				     const double* iftrue,
				     const double* iffalse,
				     const int size);




} // namespace equelleCUDA

#endif // EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED
