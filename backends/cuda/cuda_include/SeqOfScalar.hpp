
#ifndef EQUELLE_SEQOFSCALAR_HEADER_INCLUDED
#define EQUELLE_SEQOFSCALAR_HEADER_INCLUDED


#include "equelleTypedefs.hpp"

#include <vector>

namespace equelleCUDA
{

    //! Sequence of Scalar
    typedef std::vector<Scalar> SeqOfScalar;
    
    //! Multiplication: Sequence Of Scalar * Scalar
    /*!
      Needed for supporting units for variables of type Sequence Of Scalar.
     */
    SeqOfScalar operator* (const SeqOfScalar& seq, const Scalar a);

} // namespace equelleCUDA


#endif // EQUELLE_SEQOFSCALAR_HEADER_INCLUDED
