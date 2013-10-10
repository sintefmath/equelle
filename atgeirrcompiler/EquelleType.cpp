/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleType.hpp"


// ----- Implementation of utility functions -----

std::string basicTypeString(const BasicType bt)
{
    switch (bt) {
    case Bool:
        return "Bool";
    case Scalar:
        return "Scalar";
    case Vector:
        return "Vector";
    case Cell:
        return "Cell";
    case Face:
        return "Face";
    case Edge:
        return "Edge";
    case Vertex:
        return "Vertex";
    default:
        return "basicTypeString() error";
    }
}




bool isEntityType(const BasicType bt)
{
    switch (bt) {
    case Bool:
    case Scalar:
    case Vector:
    case Invalid:
        return false;
    case Cell:
    case Face:
    case Edge:
    case Vertex:
        return true;
    default:
        yyerror("internal compiler error in isEntityType().");
        return false;
    }
}




bool isNumericType(const BasicType bt)
{
    switch (bt) {
    case Bool:
    case Cell:
    case Face:
    case Edge:
    case Vertex:
    case Invalid:
        return false;
    case Scalar:
    case Vector:
        return true;
    default:
        yyerror("internal compiler error in isNumericType().");
        return false;
    }
}




std::string canonicalEntitySetString(const int gridmapping)
{
    if (gridmapping >= NotApplicable) {
        return "canonicalEntitySetString() error -- NotApplicable";
    }
    if (gridmapping < NotApplicable) {
        std::string gs;
        switch (gridmapping % 3) {
        case 0: gs += "Interior"; break;
        case 1: gs += "Boundary"; break;
        case 2: gs += "All"; break;
        default: return "canonicalEntitySetString() error";
        }
        switch (gridmapping / 3) {
        case 0: gs += "Cells"; break;
        case 1: gs += "Faces"; break;
        case 2: gs += "Edges"; break;
        case 3: gs += "Vertices"; break;
        default: return "canonicalEntitySetString() error";
        }
        return gs;
    }
    return "canonicalEntitySetString() error";
}


// ----- Implementation of class EquelleType's methods -----


EquelleType::EquelleType(const BasicType bt,
                         const bool collection,
                         const int gridmapping,
                         const int subset_of)
    : basic_type_(bt), collection_(collection), gridmapping_(gridmapping), subset_of_(subset_of)
{
}

bool EquelleType::isBasic() const
{
    return (basic_type_ != Invalid)
        && (collection_ == false)
        && (gridmapping_ == NotApplicable);
}

bool EquelleType::isEntityCollection() const
{
    return isEntityType(basic_type_) && collection_;
}

BasicType EquelleType::basicType() const
{
    return basic_type_;
}

bool EquelleType::isCollection() const
{
    return collection_;
}

int EquelleType::gridMapping() const
{
    return gridmapping_;
}

int EquelleType::subsetOf() const
{
    return subset_of_;
}

bool EquelleType::operator==(const EquelleType& et) const
{
    return basic_type_ == et.basic_type_
        && collection_ == et.collection_
        && gridmapping_ == et.gridmapping_
        && subset_of_ == et.subset_of_;
}

bool EquelleType::operator!=(const EquelleType& et) const
{
    return !operator==(et);
}



