/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLETYPE_HEADER_INCLUDED
#define EQUELLETYPE_HEADER_INCLUDED

#include "Common.hpp"

#include <string>


// ------ EquelleType class and type-related utilities ------ 

enum BasicType { Bool, Scalar, Vector, Cell, Face, Edge, Vertex, String, StencilI, StencilJ, StencilK, Void, Invalid };

enum CanonicalEntitySet { InteriorCells = 0, BoundaryCells, AllCells,
                          InteriorFaces, BoundaryFaces, AllFaces,
                          InteriorEdges, BoundaryEdges, AllEdges,
                          InteriorVertices, BoundaryVertices, AllVertices,
                          NotApplicable,
                          PostponedDefinition,
                          FirstRuntimeEntitySet };

enum CompositeType { None, Collection, Sequence };

enum { NotAnArray = -1, SomeArray = -2 };

std::string basicTypeString(const BasicType bt);

bool isEntityType(const BasicType bt);

bool isNumericType(const BasicType bt);

bool isStencilType( const BasicType bt );

BasicType canonicalGridMappingEntity(const int gridmapping);

std::string canonicalEntitySetString(const int gridmapping);


/// Encapsulates the possible types of variables in Equelle.
class EquelleType
{
public:
    EquelleType(const BasicType bt = Invalid,
                const CompositeType composite = None,
                const int gridmapping = NotApplicable,
                const int subset_of = NotApplicable,
                const bool is_mutable = false,
                const bool is_domain = false,
                const int array_size = NotAnArray,
                const bool is_stencil = false);

    bool isBasic() const;


    bool isEntityCollection() const;

    BasicType basicType() const;

    void setBasicType(const BasicType bt);

    CompositeType compositeType() const;

    void setCompositeType(CompositeType ct);

    bool isCollection() const;

    bool isDomain() const;

    bool isSequence() const;

    bool isArray() const;

    int arraySize() const;

    void setArraySize(const int array_size);

    int gridMapping() const;

    int subsetOf() const;

    bool isMutable() const;

    void setMutable(const bool is_mutable);

    bool isStencil() const;

    void setStencil(const bool is_stencil);

    bool operator==(const EquelleType& et) const;

    bool operator!=(const EquelleType& et) const;

    /// Return true if 'et' is less specific than *this,
    /// or the same.
    bool canSubstituteFor(const EquelleType& et) const;

private:
    BasicType basic_type_;
    CompositeType composite_;
    int gridmapping_;
    int subset_of_;
    bool mutable_;
    bool is_domain_;
    int array_size_;
    bool stencil_;
};



#endif // EQUELLETYPE_HEADER_INCLUDED
