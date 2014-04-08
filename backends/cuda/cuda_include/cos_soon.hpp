
#ifndef EQUELLE_COLLOFSCALAR_HEADER_INCLUDED
#define EQUELLE_COLLOFSCALAR_HEADER_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

#include "equelleTypedefs.hpp"
#include "CudaArray.hpp"


namespace equelleCUDA {

    //! Class for the Equelle CUDA Back-end
    /*!
      Class for storing and handeling CollectionOfScalar variables in Equelle.
      The class is part of the CUDA back-end of the Equelle compiler.
    */
    class CollOfScalarSoon
    {
    public:
	//! Default constructor
	CollOfScalarSoon();
	
	//! Allocating constructor
	/*! 
	  Allocates device memory for the collection. Does not initialize the collection. 
	  \param size number of scalars in the collection.
	*/
	explicit CollOfScalarSoon(const int size);
	
	//! Constructor for uniform value
	/*!
	  Allocates device memory and initialize all elements to the same value.
	  \param size Collection size
	  \param value Value assigned to each of the elements in the collection.
	*/
	explicit CollOfScalarSoon(const int size, const double value);
	
	explicit CollOfScalarSoon(const CudaArray& val);

	//! Constructor from std::vector
	/*! 
	  Used for initialize CollOfScalar when using unit tests.
	  Allocates memory and copy the vector stored on the host to the device.
	  \param host_vec Vector with the scalar values stored in host memory
	*/
	explicit CollOfScalarSoon(const std::vector<double>& host_vec);
	
	//! Copy constructor
	/*!
	  Allocates new device memory block, and makes a copy of the collection values.
	  \param coll CollOfScalar to copy from.
	*/
	CollOfScalarSoon(const CollOfScalarSoon& coll);  
	

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
	CollOfScalarSoon& operator= (const CollOfScalarSoon& other);



	//! Destructor
	/*!
	  Frees device memory as the CollOfScalarSoon goes out of scope.
	*/
	~CollOfScalarSoon();
	
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

	

	// Get a referance to the CudaArray with the actual values:
	const CudaArray& val() const;

	
	// ------------ Arithmetic operations as friends -------------
	friend CollOfScalarSoon operator + (const CollOfScalarSoon& lhs,
					    const CollOfScalarSoon& rhs);
	
    private:
	CudaArray val_;
	
    }; // class CollOfScalarSoon


} // namespace equelleCUDA

#endif // EQUELLE_COLLOFSCALAR_HEADER_INCLUDED
