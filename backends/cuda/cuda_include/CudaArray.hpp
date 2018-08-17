
#ifndef EQUELLE_CUDAARRAY_HEADER_INCLUDED
#define EQUELLE_CUDAARRAY_HEADER_INCLUDED

//#include <thrust/device_ptr.h>
//#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

//#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>


#include <string>
#include <fstream>
#include <iterator>

#include "equelleTypedefs.hpp"

//#include "DeviceGrid.hpp"
//#include "CollOfIndices.hpp"


// This is the header file for cuda!

//#include "EquelleRuntimeCUDA.hpp"


// Kernel declarations:
//
namespace equelleCUDA {
    


    //! Class for the Equelle CUDA Back-End
    /*!
      Class for storing and handeling the non-derivative part of 
      CollOfScalar variables in Equelle.

      For Equelle code that do not use NewtonSolve or NewtonSolve system,
      this class would be sufficient as a CollOfScalar.

      The class is part of the CUDA back-end of the Equelle compiler.
    */
    class CudaArray
    {
    public:
	//! Default constructor
	CudaArray();
	
	//! Allocating constructor
	/*! 
	  Allocates device memory for the collection and initialize to zero.
	  \param size number of scalars in the collection.
	*/
	explicit CudaArray(const int size);
	
	//! Constructor for uniform value
	/*!
	  Allocates device memory and initialize all elements to the same value.
	  \param size Collection size
	  \param value Value assigned to each of the elements in the collection.
	*/
	explicit CudaArray(const int size, const double value);
	
	//! Constructor from std::vector
	/*! 
	  Used for initialize CudaArray (via CollOfScalar) when using unit tests.
	  Allocates memory and copy the vector stored on the host to the device.
	  \param host_vec Vector with the scalar values stored in host memory
	*/
	explicit CudaArray(const std::vector<double>& host_vec);
	
	//! Copy constructor
	/*!
	  Allocates new device memory block, and makes a copy of the collection values.
	  \param coll cudaArray to copy from.
	*/
	CudaArray(const CudaArray& coll);  
	

	//! Copy assignment operator
	/*!
	  Overload the assignment operator. Needed for the third line here:
	  \code
	  CudaArray a = "something"
	  CudaArray b = "something"
	  a = b;
	  \endcode
	  Copy the array from other to this.
	*/
	CudaArray& operator= (const CudaArray& other);



	//! Destructor
	/*!
	  Frees device memory as the CudaArray goes out of scope.
	*/
	~CudaArray();
	
	/*! \return The size of the collection */
	int size() const;
	
	/*! 
	  \return A constant pointer to the device memory. 
	  The memory block is constant as well.
	*/
	const double* data() const;
	
	/*! \return A pointer to the device memory. */
	double* data();
	
	/*! \return A host vector containing the values of the collection */
	std::vector<double> copyToHost() const;
	
	
	//! For CUDA kernel calls.
	/*!
	  Returns a struct with the block and grid size needed to launch a
	  kernel such that we get one thread for each element in the CudaArray.
	  
	  Assumes 1D setup of grids and blocks.
	*/
	kernelSetup setup() const;



    private:
	int size_;
	double* dev_values_;
	
	// Use 1D kernel grids for arithmetic operations
	kernelSetup setup_;
	
	
	
	// Error handling
	//! check_Error throws an OPM exception if cudaStatus_ != cudaSuccess
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;
	
    
    }; // class CudaArray


    

    //! Functions closely related to the CudaArray class
    namespace wrapCudaArray {
	
	// ---------------- CUDA KERNELS ------------------- //
	

	//! Kernel for initializing to uniform values
	/*!
	  cudaMemset can only be used on 4 bytes values, and we therefore have
	  to use this kernel to initialize to uniform values.
	  \param[out] data The data array we want to initialize
	  \param[in] val The value all elements of data should get
	  \param[in] size The size of data array
	 */
	__global__ void setUniformDouble( double* data, const double val, const int size);

	//! CUDA kernel for the minus operator
	/*!
	  Performs elementwise operation for device arrays:
	  \code{.cpp} out[i] = out[i] - rhs[i] \endcode
	  \param[in,out] out Input is left hand side operand and is overwritten 
	  by the result.
	  \param[in] rhs right hand side operand.
	  \param[in] size number of elements.
	*/
	__global__ void minus_kernel(double* out, const double* rhs, const int size);
	
	//! CUDA kernel for the plus operator
	/*! 
	  Performs elementwise operation for device arrays: 
	  \code out[i] = out[i] + rhs[i] \endcode
	  \param[in,out] out Input is left hand side operand and is overwritten 
	  by the result.
	  \param[in] rhs Right hand side operand.
	  \param[in] size Number of elements.
	*/
	__global__ void plus_kernel(double* out, const double* rhs, const int size);
	
	//! CUDA kernel for the multiplication operator
	/*! 
	  Performs elementwise operation for device arrays: 
	  \code out[i] = out[i] * rhs[i] \endcode
	  \param[in,out] out Input is left hand side operand and is overwritten
	  by the result.
	  \param[in] rhs Right hand side operand.
	  \param[in] size Number of elements.
	*/
	__global__ void multiplication_kernel(double* out, 
					      const double* rhs, 
					      const int size);
	
	//! CUDA kernel for the division operator
	/*! 
	  Performs elementwise operation for device arrays: 
	  \code out[i] = out[i] / rhs[i] \endcode
	  \param[in,out] out Input is left hand side operand and is overwritten
	  by the result.
	  \param[in] rhs Right hand side operand.
	  \param[in] size Number of elements.
	*/
	__global__ void division_kernel(double* out, const double* rhs, const int size);
	
	//! CUDA kernel for multiplication with scalar and collection
	/*!
	  Multiply each element in out with the value scal.
	  \code out[i] = out[i] * scal \endcode
	  \param[in,out] out Input is the collection operand and is overwritten
	  by the result.
	  \param[in] scal Scalar value operand.
	  \param[in] size Number of elements.
	*/
	__global__ void scalMultColl_kernel(double* out, 
						  const double scal, 
						  const int size);
	
	//! CUDA kernel for division as Scalar/CudaArray
	/*!
	  Set each element in out as 
	  \code out[i] = scal/out[i] \endcode
	  \param[in,out] out Input is the denominator and is overwritten by the result.
	  \param[in] scal Scalar value numerator.
	  \param[in] size Number of elements.
	*/
	__global__ void scalDivColl_kernel( double* out,
						  const double scal,
						  const int size);
	
	//! CUDA kernel for greater than operation
	/*!
	  Compare elements in lhs with elements in rhs and return a Collection of Booleans.
	  \code out[i] = lhs[i] > rhs[i] \endcode
	  \param[in,out] out The resulting collection of booleans
	  \param[in] lhs Left hand side values
	  \param[in] rhs Right hand side values
	  \param[in] size Size of the arrays.
	*/
	__global__ void comp_collGTcoll_kernel( bool* out,
						const double* lhs,
						const double* rhs,
						const int size);
	
	//! CUDA kernel for greater than scalar operation
	/*!
	  Compare elements in lhs with a single scalar rhs and return a 
	  Collection of Booleans.
	  \code out[i] = lhs[i] > rhs \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection of scalars
	  \param[in] rhs Right hand side scalar
	  \param[in] size Size of the lhs array.
	*/
	__global__ void comp_collGTscal_kernel( bool* out,
						const double* lhs,
						const double rhs,
						const int size);

	//! CUDA kernel for greater scalar greater than collection operation
	/*!
	  Compare a single scalar lhs with elements in rhs and return a
	  Collection of Booleans.
	  \code out[i] = lhs > rhs[i] \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side scalar
	  \param[in] rhs Right hand side collection of scalars
	  \param[in] size Size of the rhs array.
	*/
	__global__ void comp_scalGTcoll_kernel( bool* out,
						const double lhs,
						const double* rhs,
						const int size);
	
	//! CUDA kernel for greater than or equal operation
	/*!
	  Compare elements in lhs with elements in rhs and return a Collection Of Booleans.
	  \code out[i] = lhs[i] >= rhs[i] \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side values
	  \param[in] rhs Right hand side values
	  \param[in] size Size of the arrays
	*/
	__global__ void comp_collGEcoll_kernel( bool* out,
						const double* lhs,
						const double* rhs,
						const int size);
	
	//! CUDA kernel for greater than or equal scalar operation
	/*!
	  Compare elements in lhs with a single scalar rhs and return a 
	  Collection of Booleans.
	  \code out[i] = lhs[i] >= rhs \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection of scalars
	  \param[in] rhs Right hand side scalar.
	  \param[in] size Size of the lhs array.
	*/
	__global__ void comp_collGEscal_kernel( bool* out,
						const double* lhs,
						const double rhs,
						const int size);
	
	//! CUDA kernel for scalar greater than or equal collection operation
	/*!
	  Compare scalar lhs with elements in rhs and return a 
	  Collection Of Booleans.
	  \code out[i] = lhs >= rhs[i] \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side scalar
	  \param[in] rhs Right hand side collection of scalar
	  \param[in] size Size of rhs.
	*/
	__global__ void comp_scalGEcoll_kernel( bool* out,
						const double lhs,
						const double* rhs,
						const int size);


	//! CUDA kernel for collection equal collection operation
	/*!
	  Compare elements in lhs with elements in rhs and return a
	  Collection Of Booleans.
	  \code out[i] = lhs[i] == rhs[i] \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection
	  \param[in] rhs Right hand side collection
	  \param[in] size Size of the collections.
	*/
	__global__ void comp_collEQcoll_kernel( bool* out,
						const double* lhs,
						const double* rhs,
						const int size);
	
	//! CUDA kernel for collection equal scalar operation
	/*!
	  Compare elements in lhs with the scalar rhs and return a
	  Collection Of Booleans.
	  \code out[i] = lhs[i] == rhs \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection
	  \param[in] rhs Right hand side scalar
	  \param[in] size Size of the collections.
	*/
	__global__ void comp_collEQscal_kernel( bool* out,
						const double* lhs,
						const double rhs,
						const int size);


	//! CUDA kernel for collection inequal collection operation
	/*!
	  Compare elements in lhs with elements in rhs and return a
	  Collection Of Booleans.
	  \code out[i] = lhs[i] != rhs[i] \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection
	  \param[in] rhs Right hand side collection
	  \param[in] size Size of the collections.
	*/
	__global__ void comp_collNEcoll_kernel( bool* out,
						const double* lhs,
						const double* rhs,
						const int size);
	
	//! CUDA kernel for collection inequal scalar operation
	/*!
	  Compare elements in lhs with the scalar rhs and return a
	  Collection Of Booleans.
	  \code out[i] = lhs[i] != rhs \endcode
	  \param[out] out The resulting collection of booleans
	  \param[in] lhs Left hand side collection
	  \param[in] rhs Right hand side scalar
	  \param[in] size Size of the collections.
	*/
	__global__ void comp_collNEscal_kernel( bool* out,
						const double* lhs,
						const double rhs,
						const int size);

    } // namespace wrapCudaArray
	
	
    // -------------- Operation overloading ------------------- //
    
    // Overloading of operator -
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs - rhs 
      \sa minus_kernel.
    */
    CudaArray operator-(const CudaArray& lhs, const CudaArray& rhs);

    // Overloading of operator +
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs + rhs 
      \sa plus_kernel.
    */
    CudaArray operator+(const CudaArray& lhs, const CudaArray& rhs);
    
    // Overloading of operator *
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs * rhs 
      \sa multiplication_kernel.
    */
    CudaArray operator*(const CudaArray& lhs, const CudaArray& rhs);

    // Overloading of operator /
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs / rhs 
      \sa division_kernel.
    */
    CudaArray operator/(const CudaArray& lhs, const CudaArray& rhs);
    

    // Multiplication:  Scalar * Collection Of Scalars
    /*!
      Wrapper for the CUDA kernel which performs the operation
      \param lhs Left hand side Scalar
      \param rhs Right hand side Collection of Scalars
      \return lhs * rhs
      \sa scalMultColl_kernel
    */
    CudaArray operator*(const Scalar lhs, const CudaArray& rhs);

    /*! 
      Since multiplication is commutative, this implementation simply return
      rhs *  lhs
      \param lhs Left hand side Collection of Scalar
      \param rhs Right han side Scalar
      \return lhs * rhs
    */
    CudaArray operator*(const CudaArray& lhs, const Scalar rhs);

    /*!
      Implemented as (1/rhs)*lhs in order to reuse kernel
      \param lhs Left hand side Collection of Scalar
      \param rhs Right hand side Scalar
      \return lhs / rhs
     */
    CudaArray operator/(const CudaArray& lhs, const Scalar rhs);

    /*!
      For Scalar / CudaArray. Elementwise division of the elements in 
      the collection.
      \param lhs Scalar 
      \param rhs Collection of Scalars
      \return out[i] = lhs / rhs[i]
     */
    CudaArray operator/(const Scalar lhs, const CudaArray& rhs);

    
    /*!
      Unary minus
      \return A collection with the negative values of the inpur collection.
    */
    CudaArray operator-(const CudaArray& arg);

    /*!
      Greater than operator
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] > rhs[i] \endcode
    */
    CollOfBool operator>(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Greater than operator comparing collection to scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] > rhs \endcode
    */
    CollOfBool operator>(const CudaArray& lhs, const Scalar rhs);

    /*!
      Greater than operator comparing scalar to collection.
      \return Collection of Booleans consisting of
      \code out[i] = lhs > rhs[i] \endcode
    */
    CollOfBool operator>(const Scalar lhs, const CudaArray& rhs);

    /*! 
      Less than operator
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] < rhs[i] \endcode
    */
    CollOfBool operator<(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Less than operator comparing with a scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] < rhs \endcode
    */
    CollOfBool operator<(const CudaArray& lhs, const Scalar rhs);

    /*!
      Less than operator comparing scalar with collection
      \return Collection of Booleans consisting of
      \code out[i] = lhs < rhs[i] \endcode
    */
    CollOfBool operator<(const Scalar lhs, const CudaArray& rhs);


    /*!
      Greater than or equal operator
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] >= rhs[i] \endcode
    */
    CollOfBool operator>=(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Greater than or equal operator comparing collection to scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] >= rhs \endcode
    */
    CollOfBool operator>=(const CudaArray& lhs, const Scalar rhs);

    /*!
      Greater than or equal operator comparing scalar to collection.
      \return Collection of Booleans consisting of
      \code out[i] = lhs >= rhs[i] \endcode
    */
    CollOfBool operator>=(const Scalar lhs, const CudaArray& rhs);


   /*!
      Less than or equal operator
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] <= rhs[i] \endcode
    */
    CollOfBool operator<=(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Less than or equal operator comparing collection to scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] <= rhs \endcode
    */
    CollOfBool operator<=(const CudaArray& lhs, const Scalar rhs);

    /*!
      Less than or equal operator comparing scalar to collection.
      \return Collection of Booleans consisting of
      \code out[i] = lhs <= rhs[i] \endcode
    */
    CollOfBool operator<=(const Scalar lhs, const CudaArray& rhs);


    // OPERATOR == 
    /*!
      Equal operator comparing two collections
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] == rhs[i] \endcode
    */
    CollOfBool operator==(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Equal operator comparing collection with scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] == rhs \endcode
    */
    CollOfBool operator==(const CudaArray& lhs, const Scalar rhs);
    
    /*!
      Equal operator comparing scalar with collection
      \return Collection of Booleans consisting of
      \code out[i] = lhs == rhs[i] \endcode
    */
    CollOfBool operator==(const Scalar lhs, const CudaArray& rhs);


    // OPERATOR !=
    /*!
      Inequal operator comparing two collections
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] != rhs[i] \endcode
    */
    CollOfBool operator!=(const CudaArray& lhs, const CudaArray& rhs);

    /*!
      Inequal operator comparing collection with scalar
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] != rhs \endcode
    */
    CollOfBool operator!=(const CudaArray& lhs, const Scalar rhs);
    
    /*!
      Inequal operator comparing scalar with collection
      \return Collection of Booleans consisting of
      \code out[i] = lhs != rhs[i] \endcode
    */
    CollOfBool operator!=(const Scalar lhs, const CudaArray& rhs);




    //! CollOfBool -> std::vector<bool>
    /*!
      Function for transforming a CollOfBool to a std::vector<bool>
    */
    std::vector<bool> cob_to_std(const CollOfBool& cob);




} // namespace equelleCUDA


#endif // EQUELLE_CUDAARRAY_HEADER_INCLUDED
