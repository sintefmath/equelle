#ifndef EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED
#define EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"


namespace equelleCUDA
{
    //! Wrapper for calling the trinaryIf kernel for CollOfScalar
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

    //! Wrapper for calling the trinaryIf kernel for CollOfIndices
    /*!
      Set up the kernel for a trinaryIf. Assumes that the given input parameters 
      satisfies the criterias as being the same size and not containing any illigal 
      values.

      \param predicate A Collection of Booleans often written as a test.
      \param iftrue Values the output elements should take if the test is true
      \param iffalse Values the output elements should take if the test is false.
      return A device vectors with elements form iftrue and iffalse according
      to predicate.
    */
    thrust::device_vector<int> trinaryIfWrapper(const CollOfBool& predicate,
						const thrust::device_vector<int>& iftrue,
						const thrust::device_vector<int>& iffalse);


    //! Kernel for trinaryIf for CollOfScalar
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

    //! Kernel for trinaryIf for CollOfIndices
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
      
      \param[out] out Result indices
      \param[in] predicate Booleans to indicate which values to assign to out.
      \param[in] iftrue Indices to be assigned to out if predicate is true
      \param[in] iffalse Indices to be assigned to out if predicate is false
      \param[in] size Size of the above arrays.
    */
    __global__ void trinaryIfKernel( int* out,
				     const bool* predicate,
				     const int* iftrue,
				     const int* iffalse,
				     const int size);



    // ----------- GRADIENT ------------------- //
    
    //! Wrapper for the Gradient kernel
    /*!
      This function provide a wrapper for calling the kernel which computes the
      gradient. For that we also need the set of interior_faces in order to know the 
      indices of the faces we want to compute the gradient of. We also need the 
      array where the information about which cells are on each side of each face.
      
      \param cell_scalarfield The input values given on AllCells.
      \param int_faces Contains the indices of the Interior Faces.
      \param face_cells The array with indices telling us which cells are on each side
      of what face.
      \return A Collection Of Scalars on all Interior Cells with the discrete gradient
      value computed from the value given in All Cells.
    */
    CollOfScalar gradientWrapper( const CollOfScalar& cell_scalarfield,
				  const CollOfFace& int_faces,
				  const int* face_cells);

    //! Kernel for computing the Gradient
    /*!
      This kernel computes the discrete gradient values on each interior face given 
      a value on all cells.
      
      \param[out] grad The resulting gradient values.
      \param[in] cell_vals The input value given as a collection on all cells.
      \param[in] interior_faces The face indices for the interior faces.
      \param[in] face_cells The array with indices telling us which cells are on each 
      side of what face.
      \param[in] size_out The size of the resulting grad array.
    */
    __global__ void gradientKernel( double* grad,
				    const double* cell_vals,
				    const int* interior_faces,
				    const int* face_cells,
				    const int size_out);


} // namespace equelleCUDA

#endif // EQUELLE_WRAP_EQUELLERUNTIME_HEADER_INCLUDED
