
#ifndef EQUELLE_COLLOFVECTOR_HEADER_INCLUDED
#define EQUELLE_COLLOFVECTOR_HEADER_INCLUDED


#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"


namespace equelleCUDA {

    class CollOfVector : public CollOfScalar 
    {
    public:
	CollOfVector();
	explicit CollOfVector(const int size, const int dim);
	explicit CollOfVector(const std::vector<double>& host, const int dim);
	CollOfVector(const CollOfVector& coll);

	CollOfScalar operator[](const int index) const;

    private:
	int dim_;

	// size_ from CollOfScalar is actually size_ * dim
    };

    __global__ void collOfVectorOperatorIndexKernel( double* out,
						     const double* vec,
						     const int size_out,
						     const int index,
						     const int dim);
	


} // namespace equelleCUDA


#endif // EQUELLE_COLLOFVECTOR_HEADER_INCLUDED
