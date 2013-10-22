/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLETYPE_HEADER_INCLUDED
#define EQUELLETYPE_HEADER_INCLUDED

#include "Common.hpp"

#include <string>


// ------ EquelleType class and type-related utilities ------ 

enum BasicType { Bool, Scalar, Vector, Cell, Face, Edge, Vertex, String, Invalid };

enum CanonicalEntitySet { InteriorCells = 0, BoundaryCells, AllCells,
                          InteriorFaces, BoundaryFaces, AllFaces,
                          InteriorEdges, BoundaryEdges, AllEdges,
                          InteriorVertices, BoundaryVertices, AllVertices,
                          NotApplicable,
                          PostponedDefinition,
                          FirstRuntimeEntitySet };

enum CompositeType { None, Collection, Sequence };

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
                const bool is_mutable = false);

    bool isBasic() const;

    bool isEntityCollection() const;

    BasicType basicType() const;

    CompositeType compositeType() const;

    bool isCollection() const;

    bool isSequence() const;

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
};



#endif // EQUELLETYPE_HEADER_INCLUDED
