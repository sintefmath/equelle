/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLETYPE_HEADER_INCLUDED
#define EQUELLETYPE_HEADER_INCLUDED

#include "Common.hpp"

#include <string>


// ------ EquelleType class and type-related utilities ------ 

enum BasicType { Bool, Scalar, Vector, Cell, Face, Edge, Vertex, String, Void, Invalid };

enum CanonicalEntitySet { InteriorCells = 0, BoundaryCells, AllCells,
                          InteriorFaces, BoundaryFaces, AllFaces,
                          InteriorEdges, BoundaryEdges, AllEdges,
                          InteriorVertices, BoundaryVertices, AllVertices,
                          NotApplicable,
                          PostponedDefinition,
                          FirstRuntimeEntitySet };

enum CompositeType { None, Collection, Sequence };

enum { NotAnArray = -1 };

std::string basicTypeString(const BasicType bt);

bool isEntityType(const BasicType bt);

bool isNumericType(const BasicType bt);

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
                const int array_size = NotAnArray);

    bool isBasic() const;

    bool isEntityCollection() const;

    BasicType basicType() const;

    void setBasicType(const BasicType bt);

    CompositeType compositeType() const;

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

    bool operator==(const EquelleType& et) const;

    bool operator!=(const EquelleType& et) const;

private:
    BasicType basic_type_;
    CompositeType composite_;
    int gridmapping_;
    int subset_of_;
    bool mutable_;
    bool is_domain_;
    int array_size_;
};



#endif // EQUELLETYPE_HEADER_INCLUDED
