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
#include <algorithm>
#include <stdexcept>


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

inline bool isEntityType(const BasicType bt)
{
    switch (bt) {
    case Bool:
    case Scalar:
    case Vector:
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

    bool isBasic() const
    {
        return (basic_type_ != Invalid)
            && (collection_ == false)
            && (gridmapping_ == NotApplicable);
    }

    bool isEntityCollection() const
    {
        return isEntityType(basic_type_) && collection_;
    }

    BasicType basicType() const
    {
        return basic_type_;
    }

    bool isCollection() const
    {
        return collection_;
    }

    int gridMapping() const
    {
        return gridmapping_;
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


class FunctionType
{
public:
    typedef std::pair<std::string, EquelleType> ArgumentType;

    FunctionType(const EquelleType return_type)
        : return_type_(return_type)
    {
    }

    FunctionType(const EquelleType return_type,
                 const std::string arg1_name, const EquelleType arg1_type)
        : return_type_(return_type),
          arguments_(1, std::make_pair(arg1_name, arg1_type))
    {
    }

    FunctionType(const EquelleType return_type,
                 const std::vector<ArgumentType>& args)
        : return_type_(return_type),
          arguments_(args)
    {
    }

    EquelleType returnType() const
    {
        return return_type_;
    }

    const std::vector<ArgumentType> arguments() const
    {
        return arguments_;
    }

private:
    EquelleType return_type_;
    std::vector<ArgumentType> arguments_;
};


class Function
{
public:
    Function(const std::string& name, const FunctionType& type)
        : name_(name),
          type_(type)
    {
    }

    void declareVariable(const std::string name, const EquelleType type)
    {
        if (!declared(name).first) {
            local_variable_types_[name] = type;
        } else {
            std::string errmsg = "redeclared variable: ";
            errmsg += name;
            yyerror(errmsg.c_str());
        }
    }

    EquelleType variableType(const std::string name) const
    {
        auto foundvar = declared(name);
        if (!foundvar.first) {
            std::string err_msg = "could not find variable ";
            err_msg += name;
            yyerror(err_msg.c_str());
            return EquelleType();
        } else {
            return foundvar.second;
        }
    }

    std::string name() const
    {
        return name_;
    }
    EquelleType returnType() const
    {
        return type_.returnType();
    }
private:
    std::pair<bool, EquelleType> declared(const std::string& name) const
    {
        auto lit = local_variable_types_.find(name);
        if (lit != local_variable_types_.end()) {
            return std::make_pair(true, lit->second);
        }
        auto ait = std::find_if(type_.arguments().begin(), type_.arguments().end(),
                                [&](const FunctionType::ArgumentType& a) { return a.first == name; });
        if (ait != type_.arguments().end()) {
            return std::make_pair(true, ait->second);
        }
        return std::make_pair(false, EquelleType());
    }
    std::string name_;
    std::map<std::string, EquelleType> local_variable_types_;
    std::vector<EntitySet> local_entitysets_;
    FunctionType type_;
};


class SymbolTable
{
public:
    static void declareVariable(const std::string& name, const EquelleType& type)
    {
        instance().current_function_->declareVariable(name, type);
    }

    static void declareFunction(const std::string& name, const FunctionType& ftype)
    {
        instance().declareFunctionImpl(name, ftype);
    }

    static const Function& getFunction(const std::string& name)
    {
        return instance().getFunctionImpl(name);
    }

    static const Function& getCurrentFunction()
    {
        return *instance().current_function_;
    }

private:
    SymbolTable()
    {
        functions_.emplace_back("InteriorCells", EquelleType(Cell, true, InteriorCells));
        functions_.emplace_back("BoundaryCells", EquelleType(Cell, true, BoundaryCells));
        functions_.emplace_back("AllCells", EquelleType(Cell, true, AllCells));
        functions_.emplace_back("InteriorFaces", EquelleType(Cell, true, InteriorFaces));
        functions_.emplace_back("BoundaryFaces", EquelleType(Cell, true, BoundaryFaces));
        functions_.emplace_back("AllFaces", EquelleType(Cell, true, AllFaces));
        functions_.emplace_back("InteriorEdges", EquelleType(Cell, true, InteriorEdges));
        functions_.emplace_back("BoundaryEdges", EquelleType(Cell, true, BoundaryEdges));
        functions_.emplace_back("AllEdges", EquelleType(Cell, true, AllEdges));
        functions_.emplace_back("InteriorVertices", EquelleType(Cell, true, InteriorVertices));
        functions_.emplace_back("BoundaryVertices", EquelleType(Cell, true, BoundaryVertices));
        functions_.emplace_back("AllVertices", EquelleType(Cell, true, AllVertices));
        functions_.emplace_back("Main", EquelleType());
        current_function_ = functions_.end() - 1;
    }

    static SymbolTable& instance()
    {
        static SymbolTable s;
        return s;
    }

    void declareFunctionImpl(const std::string& name, const FunctionType& ftype)
    {
        if (current_function_->name() != "Main") {
            std::string err_msg = "cannot define new function ";
            err_msg += name;
            err_msg += " inside function ";
            err_msg += current_function_->name();
            yyerror(err_msg.c_str());
        } else {
            auto it = findFunction(name);
            if (it == functions_.end()) {
                functions_.emplace_back(name, ftype);
            }
        }
    }

    const Function& getFunctionImpl(const std::string& name) const
    {
        auto it = findFunction(name);
        if (it == functions_.end()) {
            std::string errmsg = "could not find function ";
            errmsg += name;
            yyerror(errmsg.c_str());
            throw std::logic_error("Function not found.");
        } else {
            return *it;
        }
    }

    std::vector<Function>::const_iterator findFunction(const std::string& name) const
    {
        auto it = std::find_if(functions_.begin(), functions_.end(),
                               [&](const Function& f) { return f.name() == name; });
        return it;
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
    virtual EquelleType type() const
    {
        return EquelleType();
    }
};

typedef Node* NodePtr;

class NumberNode : public Node
{
public:
    NumberNode(const double num) : num_(num) {}
private:
    double num_;
};

class TypeNode : public Node
{
public:
    TypeNode(const EquelleType et) : et_(et) {}
    EquelleType type() const
    {
        return et_;
    }
private:
    EquelleType et_;
};

typedef TypeNode* TypeNodePtr;

class FuncTypeNode : public Node
{
public:
    FuncTypeNode(const FunctionType ft) : ft_(ft) {}
    EquelleType type() const
    {
        return ft_.returnType();
    }
    FunctionType funcType() const
    {
        return ft_;
    }
private:
    FunctionType ft_;
};

typedef FuncTypeNode* FuncTypeNodePtr;

enum BinaryOp { Add, Subtract, Multiply, Divide };

class BinaryOpNode : public Node
{
public:
    BinaryOpNode(BinaryOp op, NodePtr left, NodePtr right) : op_(op), left_(left), right_(right) {}
private:
    BinaryOp op_;
    NodePtr left_;
    NodePtr right_;
};

class NormNode : public Node
{
public:
    NormNode(NodePtr expr_to_norm) : expr_to_norm_(expr_to_norm) {}
private:
    NodePtr expr_to_norm_;
};

class UnaryNegationNode : public Node
{
public:
    UnaryNegationNode(NodePtr expr_to_negate) : expr_to_negate_(expr_to_negate) {}
private:
    NodePtr expr_to_negate_;
};

class OnNode : public Node
{
public:
    OnNode(NodePtr left, NodePtr right) : left_(left), right_(right) {}
private:
    NodePtr left_;
    NodePtr right_;
};

class TrinaryIfNode : public Node
{
public:
    TrinaryIfNode(NodePtr predicate, NodePtr iftrue, NodePtr iffalse)
        : predicate_(predicate), iftrue_(iftrue), iffalse_(iffalse)
    {}
    EquelleType type() const
    {
        return iftrue_->type();
    }
private:
    NodePtr predicate_;
    NodePtr iftrue_;
    NodePtr iffalse_;
};

class VarDeclNode : public Node
{
public:
    VarDeclNode(std::string varname, TypeNodePtr type)
        : varname_(varname), type_(type) {}
    EquelleType type() const
    {
        return type_->type();
    }
    const std::string& name() const
    {
        return varname_;
    }
private:
    std::string varname_;
    TypeNodePtr type_;
};

class VarAssignNode : public Node
{
public:
    VarAssignNode(std::string varname, NodePtr expr) : varname_(varname), expr_(expr) {}
private:
    std::string varname_;
    NodePtr expr_;
};

class VarNode : public Node
{
public:
    VarNode(const std::string& varname) : varname_(varname) {}
    EquelleType type() const
    {
        return SymbolTable::getCurrentFunction().variableType(varname_);
    }
    const std::string& name() const
    {
        return varname_;
    }
private:
    std::string varname_;
};

class FuncArgsDeclNode : public Node
{
public:
    FuncArgsDeclNode(VarDeclNode* vardecl = 0)
    {
        if (vardecl) {
            decls_.push_back(vardecl);
        }
    }
    void addArg(VarDeclNode* vardecl)
    {
        decls_.push_back(vardecl);
    }
    std::vector<FunctionType::ArgumentType> arguments() const
    {
        std::vector<FunctionType::ArgumentType> args;
        args.reserve(decls_.size());
        for (auto vdn : decls_) {
            args.push_back(std::make_pair(vdn->name(), vdn->type()));
        }
        return args;
    }
private:
    std::vector<VarDeclNode*> decls_;
};

class FuncDeclNode : public Node
{
public:
    FuncDeclNode(std::string funcname, FuncTypeNodePtr ftype)
        : funcname_(funcname), ftype_(ftype) {}
private:
    std::string funcname_;
    FuncTypeNodePtr ftype_;
};

class FuncAssignNode : public Node
{
public:
    FuncAssignNode(std::string funcname, NodePtr funcargs, NodePtr funcbody)
        : funcname_(funcname), funcargs_(funcargs), funcbody_(funcbody) {}
private:
    std::string funcname_;
    NodePtr funcargs_;
    NodePtr funcbody_;
};

class FuncCallNode : public Node
{
public:
    FuncCallNode(std::string funcname, NodePtr funcargs)
        : funcname_(funcname), funcargs_(funcargs) {}
    EquelleType type() const
    {
        return SymbolTable::getFunction(funcname_).returnType();
    }
private:
    std::string funcname_;
    NodePtr funcargs_;
};


// ------ Handle parse events ------

inline NodePtr handleNumber(const double num)
{
    return new NumberNode(num);
}

inline VarDeclNode* handleDeclaration(const std::string name, TypeNodePtr type)
{
    SymbolTable::declareVariable(name, type->type());
    return new VarDeclNode(name, type);
}

inline NodePtr handleFuncDeclaration(const std::string name, FuncTypeNodePtr ftype)
{
    SymbolTable::declareFunction(name, ftype->funcType());
    return new FuncDeclNode(name, ftype);
}

inline NodePtr handleDeclarationAssign(const std::string name, TypeNodePtr, NodePtr)
{
    if (false){//SymbolTable::functionMode()) {
    } else {
        SymbolTable::declareVariable(name, EquelleType());
    }
    return new Node();
}

inline TypeNodePtr handleCollection(TypeNodePtr btype, NodePtr gridmapping, NodePtr /*subsetof*/)
{
    EquelleType bt = btype->type();
    if (!bt.isBasic()) {
        std::string errmsg = "attempting to declare a Collection Of <nonsense>";
        yyerror(errmsg.c_str());
    }
    if (gridmapping) {
        if (!gridmapping->type().isEntityCollection() || gridmapping->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be On a Collection of Cell, Face etc.");
        }
    }
    const int gm = gridmapping ? gridmapping->type().gridMapping() : NotApplicable;
    return new TypeNode(EquelleType(bt.basicType(), true, gm));
}

inline FuncTypeNodePtr handleFuncType(FuncArgsDeclNode* argtypes, TypeNodePtr rtype)
{
    return new FuncTypeNode(FunctionType(rtype->type(), argtypes->arguments()));
}


#endif // PARSER_HEADER_INCLUDED
