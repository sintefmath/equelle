
#ifndef EQUELLE_COLLOFSCALAR_HEADER_INCLUDED
#define EQUELLE_COLLOFSCALAR_HEADER_INCLUDED

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
//! CUDA kernels for arithmetic operations on CollOfScalars
namespace equelleCUDA {
    


    //! Class for the Equelle CUDA Back-end
    /*!
      Class for storing and handeling CollectionOfScalar variables in Equelle.
      The class is part of the CUDA back-end of the Equelle compiler.
    */
    class CollOfScalar
    {
    public:
	//! Default constructor
	CollOfScalar();
	
	//! Allocating constructor
	/*! 
	  Allocates device memory for the collection. Does not initialize the collection. 
	  \param size number of scalars in the collection.
	*/
	explicit CollOfScalar(const int size);
	
	//! Constructor for uniform value
	/*!
	  Allocates device memory and initialize all elements to the same value.
	  \param size Collection size
	  \param value Value assigned to each of the elements in the collection.
	*/
	explicit CollOfScalar(const int size, const double value);
	
	//! Constructor from std::vector
	/*! 
	  Used for initialize CollOfScalar when using unit tests.
	  Allocates memory and copy the vector stored on the host to the device.
	  \param host_vec Vector with the scalar values stored in host memory
	*/
	explicit CollOfScalar(const std::vector<double>& host_vec);
	
	//! Copy constructor
	/*!
	  Allocates new device memory block, and makes a copy of the collection values.
	  \param coll Collection of Scalar to copy from.
	*/
	CollOfScalar(const CollOfScalar& coll);  
	
	//! Destructor
	/*!
	  Frees device memory as the CollOfScalar goes out of scope.
	*/
	~CollOfScalar();
	
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
	  Used for setting the CUDA grid size before calling a kernel.
	  \return The appropriate CUDA gridDim.x size 
	*/
	int grid() const;
	//! For CUDA kernel calls.
	/*!
	  Used for setting the CUDA block size before calling a kernel.
	  \return The appropriate CUDA blockDim.x size
	*/
	int block() const;
	
	
    private:
	int size_;
	double* dev_values_;
	
	// Use 1D kernel grids for arithmetic operations
	int block_x_;
	int grid_x_;
	
	
	// Error handling
	//! check_Error throws an OPM exception if cudaStatus_ != cudaSuccess
	mutable cudaError_t cudaStatus_;
	void checkError_(const std::string& msg) const;
	
	
    };

       
    // ---------------- CUDA KERNELS ------------------- //
    
    //! CUDA kernel for the minus operator
    /*!
      Performs elementwise operation for device arrays:
      \code{.cpp} out[i] = out[i] - rhs[i] \endcode
      \param[in,out] out Input is left hand side operand and is overwritten by the result.
      \param[in] rhs right hand side operand.
      \param[in] size number of elements.
    */
    __global__ void minus_kernel(double* out, const double* rhs, const int size);

    //! CUDA kernel for the plus operator
    /*! 
      Performs elementwise operation for device arrays: 
      \code out[i] = out[i] + rhs[i] \endcode
      \param[in,out] out Input is left hand side operand and is overwritten by the result.
      \param[in] rhs Right hand side operand.
      \param[in] size Number of elements.
    */
    __global__ void plus_kernel(double* out, const double* rhs, const int size);

    //! CUDA kernel for the multiplication operator
    /*! 
      Performs elementwise operation for device arrays: 
      \code out[i] = out[i] * rhs[i] \endcode
      \param[in,out] out Input is left hand side operand and is overwritten by the result.
      \param[in] rhs Right hand side operand.
      \param[in] size Number of elements.
    */
    __global__ void multiplication_kernel(double* out, const double* rhs, const int size);
    
    //! CUDA kernel for the division operator
    /*! 
      Performs elementwise operation for device arrays: 
      \code out[i] = out[i] / rhs[i] \endcode
      \param[in,out] out Input is left hand side operand and is overwritten by the result.
      \param[in] rhs Right hand side operand.
      \param[in] size Number of elements.
    */
    __global__ void division_kernel(double* out, const double* rhs, const int size);

    //! CUDA kernel for multiplication with scalar and collection
    /*!
      Multiply each element in out with the value scal.
      \code out[i] = out[i] * scal \endcode
      \param[in,out] out Input is the collection operand and is overwritten by the result.
      \param[in] scal Scalar value operand.
      \param[in] size Number of elements.
    */
    __global__ void multScalCollection_kernel(double* out, 
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
    __global__ void compGTkernel( bool* out,
				  const double* lhs,
				  const double* rhs,
				  const int size);

    // -------------- Operation overloading ------------------- //
    
    // Overloading of operator -
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs - rhs 
      \sa minus_kernel.
    */
    CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs);

    // Overloading of operator +
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs + rhs 
      \sa plus_kernel.
    */
    CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs);
    
    // Overloading of operator *
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs * rhs 
      \sa multiplication_kernel.
    */
    CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs);

    // Overloading of operator /
    /*!
      Wrapper for the CUDA kernel which performs the operation.
      \param lhs Left hand side operand
      \param rhs Right hand side operand
      \return lhs / rhs 
      \sa division_kernel.
    */
    CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs);
    

    // Multiplication:  Scalar * Collection Of Scalars
    /*!
      Wrapper for the CUDA kernel which performs the operation
      \param lhs Left hand side Scalar
      \param rhs Right hand side Collection of Scalars
      \return lhs * rhs
      \sa multScalCollection_kernel
    */
    CollOfScalar operator*(const Scalar& lhs, const CollOfScalar& rhs);

    /*! 
      Since multiplication is commutative, this implementation simply return
      rhs *  lhs
      \param lhs Left hand side Collection of Scalar
      \param rhs Right han side Scalar
      \return lhs * rhs
    */
    CollOfScalar operator*(const CollOfScalar& lhs, const Scalar& rhs);

    /*!
      Implemented as (1/rhs)*lhs in order to reuse kernel
      \param lhs Left hand side Collection of Scalar
      \param rhs Right hand side Scalar
      \return lhs / rhs
     */
    CollOfScalar operator/(const CollOfScalar& lhs, const Scalar& rhs);

    /*!
      Unary minus
      \return A collection with the negative values of the inpur collection.
    */
    CollOfScalar operator-(const CollOfScalar& arg);

    /*!
      Greater than operator
      \return Collection of Booleans consisting of
      \code out[i] = lhs[i] > rhs[i] \endcode
     */
    CollOfBool operator>(const CollOfScalar& lhs, const CollOfScalar& rhs);



    //! CollOfBool -> std::vector<bool>
    /*!
      Function for transforming a CollOfBool to a std::vector<bool>
    */
    std::vector<bool> cob_to_std(const CollOfBool& cob);

    /*!
      Define max number of threads in a kernel block:
    */
    const int MAX_THREADS = 512;
    
} // namespace equelleCUDA


#endif // EQUELLE_COLLOFSCALAR_CUDA_HEADER_INCLUDED
