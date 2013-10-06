/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLETYPES_HEADER_INCLUDED
#define EQUELLETYPES_HEADER_INCLUDED

enum BasicTypes { Boolean, Scalar, Vector, Cell, Face, Edge, Vertex };



#include <string>

struct EquelleType
{
    virtual std::string equelleDeclaration() const = 0;
    virtual std::string backendDeclaration() const = 0;
};

struct Entity : public EquelleType
{
    std::string ename;
    std::string equelleDeclaration() const {
        return ename;
    }
    std::string backendDeclaration() const {
        return ename;
    }
};

struct Cell : public Entity {};
struct Face : public Entity {};
struct Edge : public Entity {};
struct Vertex : public Entity {};

struct Numeric : public EquelleType {};

struct Scalar : public Numeric {};
struct Vector : public Numeric {};

struct Boolean : public EquelleType {};

struct Collection : public EquelleType
{
    const std::unique_ptr<EquelleType> contained_type;
    std::string equelleDeclaration() const {
        std::string rval = "Collection Of ";
        rval += contained_type.equelleDeclaration();
        return rval;
    }
    std::string backendDeclaration() const {
        std::string rval = "CollOf";
        rval += contained_type.equelleDeclaration();
        return rval;
    }
};

#endif // EQUELLETYPES_HEADER_INCLUDED
