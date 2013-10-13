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
                const bool collection = false,
                const int gridmapping = NotApplicable,
                const int subset_of = NotApplicable);

    bool isBasic() const;

    bool isEntityCollection() const;

    BasicType basicType() const;

    bool isCollection() const;

    int gridMapping() const;

    int subsetOf() const;

    bool operator==(const EquelleType& et) const;

    bool operator!=(const EquelleType& et) const;

private:
    BasicType basic_type_;
    bool collection_;
    int gridmapping_;
    int subset_of_;
};



#endif // EQUELLETYPE_HEADER_INCLUDED
