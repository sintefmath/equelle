/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSER_HEADER_INCLUDED
#define PARSER_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <vector>


// ------ Declarations needed for bison parser ------ 

void yyerror(const char* s);
extern int yylex();
extern int yylineno;




// ------ Utilities used in bison parser ------ 

inline double numFromString(const std::string& s)
{
    std::istringstream iss(s);
    double num;
    iss >> num;
    if (!iss) {
        yyerror("error in string-to-number conversion.");
    }
    return num;
}




// ------ Equelle types and related utilities ------ 

enum BasicType { Bool, Scalar, Vector, Cell, Face, Edge, Vertex, Invalid };

inline std::string basicTypeString(const BasicType bt)
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

enum CanonicalEntitySet { InteriorCells = 0, BoundaryCells, AllCells,
                          InteriorFaces, BoundaryFaces, AllFaces,
                          InteriorEdges, BoundaryEdges, AllEdges,
                          InteriorVertices, BoundaryVertices, AllVertices,
                          NotApplicable,
                          FirstRuntimeEntitySet };

inline std::string canonicalEntitySetString(const int gridmapping)
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
}

class EquelleType
{
public:
    EquelleType(const BasicType bt = Invalid,
                const bool collection = false,
                const int gridmapping = NotApplicable)
        : basic_type_(bt), collection_(collection), gridmapping_(gridmapping)
    {
    }

    std::string equelleString() const
    {
        std::string es = collection_ ? "Collection Of " : "";
        es += basicTypeString(basic_type_);
        if (gridmapping_ != NotApplicable) {
            es += " On ";
            // es += SymbolTable::instance().entitySetName(gridmapping_);
        }
        return es;
    }

    std::string backendString() const
    {
        return "No backend string yet.";
    }

    bool operator==(const EquelleType& et) const
    {
        return basic_type_ == et.basic_type_
            && collection_ == et.collection_
            && gridmapping_ == et.gridmapping_;
    }

private:
    BasicType basic_type_;
    bool collection_;
    int gridmapping_;
};




// ------ SymbolTable singleton ------ 

class EntitySet
{
};


class Function
{
public:
    void declareVariable(const std::string name, const EquelleType type)
    {
        if (variable_types_.find(name) == variable_types_.end()) {
            variable_types_[name] = type;
        } else {
            std::string errmsg = "redeclared variable: ";
            errmsg += name;
            yyerror(errmsg.c_str());
        }
    }
private:
    std::string name_;
    std::map<std::string, EquelleType> variable_types_;
    std::vector<EntitySet> entitysets_;
};


class SymbolTable
{
public:
    static void declareVariable(const std::string name, const EquelleType type)
    {
        instance().current_function_->declareVariable(name, type);
    }

private:
    SymbolTable()
        : functions_(1),
          current_function_(functions_.begin())
    {
    }

    static SymbolTable& instance()
    {
        static SymbolTable s;
        return s;
    }

    std::vector<Function> functions_;
    std::vector<Function>::iterator current_function_;
};




// ------ Abstract syntax tree classes ------ 

class Node
{
public:
    virtual ~Node()
    {}
};

typedef Node* NodePtr;

class Number : public Node
{
public:
    Number(const double num) : num_(num) {}
private:
    double num_;
};




// ------ Handle parse events ------

inline NodePtr handleNumber(const double num)
{
    return new Number(num);
}

inline NodePtr handleDeclaration(const std::string name, NodePtr)
{
    if (false){//SymbolTable::functionMode()) {
    } else {
        SymbolTable::declareVariable(name, EquelleType());
    }
    return new Node();
}


#endif // PARSER_HEADER_INCLUDED
