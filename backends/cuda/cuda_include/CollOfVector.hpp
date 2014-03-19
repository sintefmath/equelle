
#ifndef EQUELLE_COLLOFVECTOR_HEADER_INCLUDED
#define EQUELLE_COLLOFVECTOR_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"


namespace equelleCUDA {

    //! Collection of Vectors
    /*!
      This class is implemented as an extention of the CollOfScalar class. It has all the
      same functionality, and is extended with a index operator which returns a
      CollOfScalar. When talking about the dimension of the collection we are really
      talking about the dimension of the vectors in the collection. The dimension is 
      therefore ment ot be 2 or 3 corresponding to 2D or 3D grids.
    */
    class CollOfVector : public CollOfScalar 
    {
    public:
	//! Default constructor
	CollOfVector();

	//! Allocating constructor without dim -> throws an error
	/*!
	  Since we cannot create a vector without stating its dimension,
	  calling this constructor will result in an error.
	*/
	explicit CollOfVector(const int size);
	
	//! Allocating constructor
	/*!
	  Allocates device memory without initialization.
	  \param size The number of vectors in the collection
	  \param dim Dimension of each vector
	*/
	explicit CollOfVector(const int size, const int dim);

	//! Constructor from std::vector
	/*!
	  Used for easy testing. The std::vectors contains the vector elements,
	  not the vectors them selves. The size of the collection will therefore
	  be host.size()/dim.
	  \param host A host vector with {1_x, 1_y, 1_z, 2_x, 2_y, 2_x,...,N_x, 
	  N_y, N_z} for a 3 dimensional case. 
	  \param dim The dimension of the vectors stored in host.
	*/
	explicit CollOfVector(const std::vector<double>& host, const int dim);

	//! Copy constructor
	/*!
	  Allocates memory for a new CollOfVector and copies all data from coll 
	  over to the new variable.
	*/
	CollOfVector(const CollOfVector& coll);

	//! Copy assignment operator
	/*!
	  Overload the assignment operator to ensure correct behaviour when 
	  we assign a CollOfVector to a CollOfVector that is already initialized.
	*/
	CollOfVector& operator= (const CollOfVector& other);

	//! Destructor
	/*!
	  Needed in order to automatically call the base class destructor.
	*/
	~CollOfVector();
	


	//! Norm of the vectors in the collection
	/*!
	  Returns a collection of scalars equal to the norm of every vector in
	  the caller. The norm used here is the 2-norm equal to the square root of 
	  the sum of the squared elements of each vector.
	*/
	CollOfScalar norm() const;

	
	//! Index operator
	/*!
	  Returns a collection of Scalars with the values from the index of each of the
	  vectors. myVector[1] will not return the second vector in the collection
	  but a collection of the second component from all the vectors.
	*/
	CollOfScalar operator[](const int index) const;

	//! Dimension of vectors in the collection
	int dim() const;

	//! Number of vectors in the collection
	/*!
	  This function returns the number of vectors in the collection.
	  Not to be confused with size() which returns total number of 
	  elements in the collection.
	  size() = numVectors()*dim()
	*/
	int numVectors() const;

    private:
	const int dim_;

	// size_ from CollOfScalar is actually size_ * dim
	// block() and grid() will therefore be evaluated as one thread per double
    };

    //! Kernel for getting the index element of all vectors in a collection.
    /*!
      \param[out] out Collection Of Scalar where out[i] is the index element of
      vector number i in the collection of Vectors
      \param[in] vec Collection of Vectors as a double array of size size_out*dim
      \param[in] size_out Number of vectors in the collection and also the number 
      of elements in the output collection of scalars.
      \param[in] index The index we want to read from each vector
      \param[in] dim The dimension of the vectors in the collection.
    */
    __global__ void collOfVectorOperatorIndexKernel( double* out,
						     const double* vec,
						     const int size_out,
						     const int index,
						     const int dim);
	
    //! Kernel for computing the norm of vectors
    /*!
      Uses one thread for each vector to compute the given vectors norm.
      
      \param[out] out The output with the norm of each vector
      \param[in] vectors Array with vector elements so that each vector is 
      continously in memory. The size of this array is numVectors*dim.
      \param[in] numVectors Number of vectors given in input
      \param[in] dim Dimension of each vector.
    */
    __global__ void normKernel( double* out,
				const double* vectors,
				const int numVectors,
				const int dim);
    


    // --------------------- OPERATOR OVERLOADING -------------------------

    /*!
      Overloaded operator + for Collection of Vectors. Elementwise addition
      of all values stored in the collections.
      
      Works as a wrapper for the CUDA kernel which add collection of scalars.
    */
    CollOfVector operator+(const CollOfVector& lhs, const CollOfVector& rhs);

    /*!
      Overloaded operator - for Collection of Vectors. Elementwise subtraction
      of all values stored in the collection.

      Works as a wrapper for the CUDA kernel which subtract collection of scalars.
    */
    CollOfVector operator-(const CollOfVector& lhs, const CollOfVector& rhs);

} // namespace equelleCUDA


#endif // EQUELLE_COLLOFVECTOR_HEADER_INCLUDED
