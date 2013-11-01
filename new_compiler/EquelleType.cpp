/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleType.hpp"
#include <sstream>

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
    case String:
        return "String";
    case Void:
        return "Void";
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
    case String:
    case Void:
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
    case String:
    case Void:
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
    if (gridmapping >= FirstRuntimeEntitySet) {
        const int index_of_set = gridmapping - FirstRuntimeEntitySet;
        std::ostringstream oss;
        oss << "RuntimeEntityset<" << index_of_set << ">";
        return oss.str();
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
    if (gridmapping == NotApplicable) {
        return "<NotApplicable>";
    }
    if (gridmapping == PostponedDefinition) {
        return "<PostponedDefinition>";
    }
    return "<Error in canonicalEntitySetString>";
}



BasicType canonicalGridMappingEntity(const int gridmapping)
{
    switch (gridmapping) {
    case InteriorCells:
    case BoundaryCells:
    case AllCells:
        return Cell;
    case InteriorFaces:
    case BoundaryFaces:
    case AllFaces:
        return Face;
    case InteriorEdges:
    case BoundaryEdges:
    case AllEdges:
        return Edge;
    case InteriorVertices:
    case BoundaryVertices:
    case AllVertices:
        return Vertex;
    default:
        return Invalid;
    }
}




// ----- Implementation of class EquelleType's methods -----


EquelleType::EquelleType(const BasicType bt,
                         const CompositeType composite,
                         const int gridmapping,
                         const int subset_of,
                         const bool is_mutable,
                         const bool is_domain)
    : basic_type_(bt),
      composite_(composite),
      gridmapping_(gridmapping),
      subset_of_(subset_of),
      mutable_(is_mutable),
      is_domain_(is_domain)
{
}

bool EquelleType::isBasic() const
{
    return (basic_type_ != Invalid)
        && (composite_ == None)
        && (gridmapping_ == NotApplicable);
}

bool EquelleType::isEntityCollection() const
{
    return isEntityType(basic_type_) && composite_ == Collection;
}

BasicType EquelleType::basicType() const
{
    return basic_type_;
}

void EquelleType::setBasicType(const BasicType bt)
{
    basic_type_ = bt;
}

CompositeType EquelleType::compositeType() const
{
    return composite_;
}

bool EquelleType::isCollection() const
{
    return composite_ == Collection;
}

bool EquelleType::isDomain() const
{
    return is_domain_;
}

bool EquelleType::isSequence() const
{
    return composite_ == Sequence;
}

int EquelleType::gridMapping() const
{
    return gridmapping_;
}

int EquelleType::subsetOf() const
{
    return subset_of_;
}

bool EquelleType::isMutable() const
{
    return mutable_;
}

void EquelleType::setMutable(const bool is_mutable)
{
    mutable_ = is_mutable;
}

bool EquelleType::operator==(const EquelleType& et) const
{
    // Note that we explicitly keep mutable_ out
    // of the equality consideration. That must be
    // checked by logic elsewhere.
    return basic_type_ == et.basic_type_
        && composite_ == et.composite_
        && gridmapping_ == et.gridmapping_
        && subset_of_ == et.subset_of_
        && is_domain_ == et.is_domain_;
}

bool EquelleType::operator!=(const EquelleType& et) const
{
    return !operator==(et);
}



