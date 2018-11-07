
#ifndef EQUELLE_COLLOFSCALAR_HEADER_INCLUDED
#define EQUELLE_COLLOFSCALAR_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include "equelleTypedefs.hpp"
#include "CudaArray.hpp"
#include "CudaMatrix.hpp"
#include "wrapDeviceGrid.hpp"


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
	  Creates an uninitialized CudaArray, by calling its allocation constructor
	  \param size number of scalars in the collection.
	*/
	explicit CollOfScalar(const int size);
	
	//! Constructor for uniform value
	/*!
	  Create a CudaArray of given size with uniform value.
	  \param size Collection size
	  \param value Value assigned to each of the elements in the collection.
	*/
	explicit CollOfScalar(const int size, const double value);
	
	//! Constructor from CudaArray
	/*!
	  Copy the given CudaArray.
	  \param val The CudaArray we want to create a CollOfScalar from.
	*/
	explicit CollOfScalar(const CudaArray& val);

	// Move constructor which moves CudaArray into val_
	explicit CollOfScalar(CudaArray&& val);

	//! Constructor for creating a primary variable
	/*!
	  Input should be a non-autodiff CollOfScalar, and will be copied to
	  this new CollOfScalar. In addition, we create an identity matrix 
	  for the derivative and all operations based on this variable will
	  be using Automatic Differentiation.
	*/
	explicit CollOfScalar(const CollOfScalar& val, const bool primaryVariable);

	//! Constructor from CudaArray and CudaMatrix.
	/*!
	  Copy the given CudaArray and CudaMatrix into a CollOfScalar, as is 
	  required when we return CollOfScalars from arithmetic expressions or
	  other operations.
	*/
	explicit CollOfScalar(const CudaArray& val, const CudaMatrix& der);

	// Move constructor from CudaArray and CudaMatrix.
	// Both val and der are moved.
	CollOfScalar(CudaArray&& val, CudaMatrix&& der) noexcept;

	// Move constructor from CudaArray and CudaMatrix
	// Only der is moved. val is copied.
	CollOfScalar(const CudaArray& val, CudaMatrix&& der);

	// Move constructor from CudaArray and CudaMatrix
	// Only der is moved. val is copied.
	CollOfScalar(CudaArray&& val, const CudaMatrix& der);

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
	  \param coll CollOfScalar to copy from.
	*/
	CollOfScalar(const CollOfScalar& coll);  
	
	// Move constructor
	CollOfScalar(CollOfScalar&& coll);

	//! Copy assignment operator
	/*!
	  Overload the assignment operator. Needed for the third line here:
	  \code
	  CollOfScalar a = "something"
	  CollOfScalar b = "something"
	  a = b;
	  \endcode
	  Copy the array from other to this.
	*/
	CollOfScalar& operator= (const CollOfScalar& other);

	// Move assignment operator
	CollOfScalar& operator=(CollOfScalar&& other);

	// Move compound operators
	CollOfScalar& operator*=(const Scalar lhs);
	CollOfScalar& operator*=(const CollOfScalar& rhs);

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
	/*! \return A hostMat struct of the derivative stored on host */
	hostMat matrixToHost() const;
	
	/*! \return True if the derivative matrix is non-empty */
	bool useAutoDiff() const;
	
	//! For CUDA kernel calls.
	/*!
	  Returns a struct with the block and grid size needed to launch a
	  kernel such that we get one thread for each element in the CudaArray.
	  
	  Assumes 1D setup of grids and blocks.
	*/
	kernelSetup setup() const;

	//! Returns a copy of the derivative matrix
	/*!
	  If the CollOfScalar do not use AutoDiff, then it returns an empty matrix. 
	*/
	CudaMatrix derivative() const;

	//! Returns a copy of the values of the CudaArray.
	CudaArray value() const;
	
	//! Reduction function
	/*!
	  This function takes care of all reduction operations in Equelle, and which 
	  operation to do is given by the reduce parameter.
	  
	  It is implemented by using thrust algorithms that relays on thrust iterators.
	  We therefore do a suboptimal copy of the data over to a 
	  thrust::device_vector.
	*/
	double reduce(const EquelleReduce reduce) const;

	// Get a referance to the CudaArray with the actual values:
	// const CudaArray& val() const;

	
	// ------------ Arithmetic operations as friends -----------------

	// Overloading of operator -
	/*!
	  Wrapper for the CUDA kernel which performs the operation.
	  \param lhs Left hand side operand
	  \param rhs Right hand side operand
	  \return lhs - rhs 
	  \sa minus_kernel.
	*/
	friend CollOfScalar operator-(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	// Overloading of operator +
	/*!
	  Wrapper for the CUDA kernel which performs the operation.
	  \param lhs Left hand side operand
	  \param rhs Right hand side operand
	  \return lhs + rhs 
	  \sa plus_kernel.
	*/
	friend CollOfScalar operator+(const CollOfScalar& lhs, const CollOfScalar& rhs);
    
	// Overloading of operator *
	/*!
	  Wrapper for the CUDA kernel which performs the operation.
	  \param lhs Left hand side operand
	  \param rhs Right hand side operand
	  \return lhs * rhs 
	  \sa multiplication_kernel.
	*/
	friend CollOfScalar operator*(const CollOfScalar& lhs, const CollOfScalar& rhs);

	// Move multiplication
	friend CollOfScalar operator*(CollOfScalar&& lhs, CollOfScalar&& rhs);

	// Overloading of operator /
	/*!
	  Wrapper for the CUDA kernel which performs the operation.
	  \param lhs Left hand side operand
	  \param rhs Right hand side operand
	  \return lhs / rhs 
	  \sa division_kernel.
	*/
	friend CollOfScalar operator/(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	// Move division operator
	friend CollOfScalar operator/(CollOfScalar&& lhs, CollOfScalar&& rhs);
	
	// Multiplication:  Scalar * Collection Of Scalars
	/*!
	  Wrapper for the CUDA kernel which performs the operation
	  \param lhs Left hand side Scalar
	  \param rhs Right hand side Collection of Scalars
	  \return lhs * rhs
	  \sa scalMultColl_kernel
	*/
	friend CollOfScalar operator*(const Scalar lhs, const CollOfScalar& rhs);

	// Move division
	friend CollOfScalar operator/(const Scalar lhs, CollOfScalar&& rhs);

	/*! 
	  Since multiplication is commutative, this implementation simply return
	  rhs *  lhs
	  \param lhs Left hand side Collection of Scalar
	  \param rhs Right han side Scalar
	  \return lhs * rhs
	*/
	friend CollOfScalar operator*(const CollOfScalar& lhs, const Scalar rhs);
	friend CollOfScalar operator*(const Scalar lhs, CollOfScalar&& rhs);

	/*!
	  Implemented as (1/rhs)*lhs in order to reuse kernel
	  \param lhs Left hand side Collection of Scalar
	  \param rhs Right hand side Scalar
	  \return lhs / rhs
	*/
	friend CollOfScalar operator/(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  For Scalar / CollOfScalar. Elementwise division of the elements in 
	  the collection.
	  \param lhs Scalar 
	  \param rhs Collection of Scalars
	  \return out[i] = lhs / rhs[i]
	*/
	friend CollOfScalar operator/(const Scalar lhs, const CollOfScalar& rhs);
	
	
	/*!
	  Unary minus
	  \return A collection with the negative values of the inpur collection.
	*/
	friend CollOfScalar operator-(const CollOfScalar& arg);
	

	// ----------- Arithmetic operations as friends end ---------------

	// ----------- Boolean operations as friends --- ---------------

	/*!
	  Greater than operator
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] > rhs[i] \endcode
	*/
	friend CollOfBool operator>(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Greater than operator comparing collection to scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] > rhs \endcode
	*/
	friend CollOfBool operator>(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Greater than operator comparing scalar to collection.
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs > rhs[i] \endcode
	*/
	friend CollOfBool operator>(const Scalar lhs, const CollOfScalar& rhs);
	
	/*! 
	  Less than operator
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] < rhs[i] \endcode
	*/
	friend CollOfBool operator<(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Less than operator comparing with a scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] < rhs \endcode
	*/
	friend CollOfBool operator<(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Less than operator comparing scalar with collection
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs < rhs[i] \endcode
	*/
	friend CollOfBool operator<(const Scalar lhs, const CollOfScalar& rhs);
	
	
	/*!
	  Greater than or equal operator
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] >= rhs[i] \endcode
	*/
	friend CollOfBool operator>=(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Greater than or equal operator comparing collection to scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] >= rhs \endcode
	*/
	friend CollOfBool operator>=(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Greater than or equal operator comparing scalar to collection.
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs >= rhs[i] \endcode
	*/
	friend CollOfBool operator>=(const Scalar lhs, const CollOfScalar& rhs);
	
	
	/*!
	  Less than or equal operator
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] <= rhs[i] \endcode
	*/
	friend CollOfBool operator<=(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Less than or equal operator comparing collection to scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] <= rhs \endcode
	*/
	friend CollOfBool operator<=(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Less than or equal operator comparing scalar to collection.
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs <= rhs[i] \endcode
	*/
	friend CollOfBool operator<=(const Scalar lhs, const CollOfScalar& rhs);
	
	
	// OPERATOR == 
	/*!
	  Equal operator comparing two collections
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] == rhs[i] \endcode
	*/
	friend CollOfBool operator==(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Equal operator comparing collection with scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] == rhs \endcode
	*/
	friend CollOfBool operator==(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Equal operator comparing scalar with collection
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs == rhs[i] \endcode
	*/
	friend CollOfBool operator==(const Scalar lhs, const CollOfScalar& rhs);
	
	
	// OPERATOR !=
	/*!
	  Inequal operator comparing two collections
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] != rhs[i] \endcode
	*/
	friend CollOfBool operator!=(const CollOfScalar& lhs, const CollOfScalar& rhs);
	
	/*!
	  Inequal operator comparing collection with scalar
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs[i] != rhs \endcode
	*/
	friend CollOfBool operator!=(const CollOfScalar& lhs, const Scalar rhs);
	
	/*!
	  Inequal operator comparing scalar with collection
	  \return Collection of Booleans consisting of
	  \code out[i] = lhs != rhs[i] \endcode
	*/
	friend CollOfBool operator!=(const Scalar lhs, const CollOfScalar& rhs);


	friend CollOfScalar wrapDeviceGrid::extendToFull( const CollOfScalar& in_data,
                       const thrust::device_vector<int>& from_set,
                       const int full_size);

	// ----------- Boolean operations as friends end ---------------
	
    private:
	CudaArray val_;
	CudaMatrix der_;
	bool autodiff_;
	
    }; // class CollOfScalar



    //! Matrix * CollOfScalar operator
    /*! 
      Many Equelle operations are used by multiplying with a matrix, and by overloading
      the operator to operate on a CollOfScalar we avoid checking if we have to 
      use AutoDiff in several functions.

      This makes it easier to implement new functionality without having to 
      think about the derivatives.
    */
    CollOfScalar operator*(const CudaMatrix& mat, const CollOfScalar& coll);
    
    


} // namespace equelleCUDA

#endif // EQUELLE_COLLOFSCALAR_HEADER_INCLUDED
