/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSER_HEADER_INCLUDED
#define PARSER_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <algorithm>
#include <stdexcept>
#include <cassert>


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

inline bool isNumericType(const BasicType bt)
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

enum CanonicalEntitySet { InteriorCells = 0, BoundaryCells, AllCells,
                          InteriorFaces, BoundaryFaces, AllFaces,
                          InteriorEdges, BoundaryEdges, AllEdges,
                          InteriorVertices, BoundaryVertices, AllVertices,
                          NotApplicable,
                          PostponedDefinition,
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
                const int gridmapping = NotApplicable,
                const int subset_of = NotApplicable)
        : basic_type_(bt), collection_(collection), gridmapping_(gridmapping), subset_of_(subset_of)
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

    int subsetOf() const
    {
        return subset_of_;
    }

    bool operator==(const EquelleType& et) const
    {
        return basic_type_ == et.basic_type_
            && collection_ == et.collection_
            && gridmapping_ == et.gridmapping_
            && subset_of_ == et.subset_of_;
    }

    bool operator!=(const EquelleType& et) const
    {
        return !operator==(et);
    }

private:
    BasicType basic_type_;
    bool collection_;
    int gridmapping_;
    int subset_of_;
};




// ------ SymbolTable singleton and classes used by it. ------ 



class EntitySet
{
public:
    EntitySet(const int index, const int subset_index)
        : index_(index), subset_index_(subset_index)
    {
    }
    int index() const
    {
        return index_;
    }
    int subsetIndex() const
    {
        return subset_index_;
    }

private:
    int index_;
    int subset_index_;
};


class Variable
{
public:
    Variable(const std::string& name,
             const EquelleType type = EquelleType(),
             bool assigned = false)
        : name_(name), type_(type), assigned_(assigned)
    {
    }
    const std::string& name() const
    {
        return name_;
    }
    const EquelleType& type() const
    {
        return type_;
    }
    void setType(const EquelleType& type)
    {
        type_ = type;
    }
    bool assigned() const
    {
        return assigned_;
    }
    void setAssigned(const bool assigned)
    {
        assigned_ = assigned;
    }
    bool operator<(const Variable& v) const
    {
        return name_ < v.name_;
    }
private:
    std::string name_;
    EquelleType type_;
    bool assigned_;
};

enum { InvalidIndex = -1 };

struct DynamicReturnSpecification
{
    DynamicReturnSpecification()
        : active(false),
          arg_index_for_basic_type(InvalidIndex),
          arg_index_for_gridmapping(InvalidIndex),
          arg_index_for_subset(InvalidIndex)
    {
    }
    DynamicReturnSpecification(const int bt_ix, const int gm_ix, const int ss_ix)
        : active(true),
          arg_index_for_basic_type(bt_ix),
          arg_index_for_gridmapping(gm_ix),
          arg_index_for_subset(ss_ix)
    {
    }
    bool active;
    int arg_index_for_basic_type;
    int arg_index_for_gridmapping;
    int arg_index_for_subset;
};


class FunctionType
{
public:
    /// Construct FunctionType taking no arguments.
    /// Equelle type: Function() -> returntype
    explicit FunctionType(const EquelleType& return_type)
        : return_type_(return_type)
    {
    }

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type)
        : arguments_(args),
          return_type_(return_type)
    {
    }

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type,
                 const DynamicReturnSpecification& dynamic)
        : arguments_(args),
          return_type_(return_type),
          dynamic_(dynamic)
    {
    }

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const
    {
        if (dynamic_.active) {
            const BasicType bt = dynamic_.arg_index_for_basic_type == InvalidIndex ?
                return_type_.basicType() : argtypes[dynamic_.arg_index_for_basic_type].basicType();
            const bool coll = return_type_.isCollection();
            const int gridmapping = dynamic_.arg_index_for_gridmapping == InvalidIndex ?
                return_type_.gridMapping() : argtypes[dynamic_.arg_index_for_gridmapping].gridMapping();
            return EquelleType(bt, coll, gridmapping);
        } else {
            return return_type_;
        }
    }

    int dynamicSubsetReturn(const std::vector<EquelleType>& argtypes) const
    {
        if (dynamic_.active) {
            const BasicType bt = dynamic_.arg_index_for_basic_type == InvalidIndex ?
                return_type_.basicType() : argtypes[dynamic_.arg_index_for_basic_type].basicType();
            const bool coll = return_type_.isCollection();
            if (isEntityType(bt) && coll) {
                const int subset = dynamic_.arg_index_for_subset == InvalidIndex ?
                    NotApplicable : argtypes[dynamic_.arg_index_for_subset].gridMapping();
                return subset;
            }
        }
        return NotApplicable;
    }

    const std::vector<Variable>& arguments() const
    {
        return arguments_;
    }

private:
    std::vector<Variable> arguments_;
    EquelleType return_type_;
    DynamicReturnSpecification dynamic_;
};


class Function
{
public:
    Function(const std::string& name, const FunctionType& type)
        : name_(name),
          type_(type)
    {
    }

    void declareVariable(const std::string& name, const EquelleType& type)
    {
        if (!declared(name).first) {
            local_variables_.insert(Variable(name, type, false));
        } else {
            std::string errmsg = "redeclared variable: ";
            errmsg += name;
            yyerror(errmsg.c_str());
        }
    }

    int declareEntitySet(const int new_entity_index, const int subset_entity_index)
    {
        local_entitysets_.emplace_back(new_entity_index, subset_entity_index);
        return new_entity_index;
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

    bool isVariableDeclared(const std::string& name) const
    {
        return declared(name).first;
    }

    bool isVariableAssigned(const std::string& name) const
    {
        auto lit = local_variables_.find(Variable(name));
        if (lit != local_variables_.end()) {
            return lit->assigned();
        } else {
            auto ait = std::find_if(type_.arguments().begin(), type_.arguments().end(),
                                    [&](const Variable& a) { return a.name() == name; });
            if (ait != type_.arguments().end()) {
                return ait->assigned();
            } else {
                yyerror("internal compiler error in Function::isVariableAssigned()");
                return false;
            }
        }
    }

    void setVariableAssigned(const std::string& name, const bool assigned)
    {
        auto lit = local_variables_.find(Variable(name));
        if (lit != local_variables_.end()) {
            // Set members are immutable, must
            // copy, erase and reinsert.
            Variable copy = *lit;
            copy.setAssigned(assigned);
            local_variables_.erase(lit);
            local_variables_.insert(copy);
        } else {
            yyerror("internal compiler error in Function::setVariableAssigned()");
        }
    }

    void setVariableType(const std::string& name, const EquelleType& type)
    {
        auto lit = local_variables_.find(Variable(name));
        if (lit != local_variables_.end()) {
            // Set members are immutable, must
            // copy, erase and reinsert.
            Variable copy = *lit;
            copy.setType(type);
            local_variables_.erase(lit);
            local_variables_.insert(copy);
        } else {
            yyerror("internal compiler error in Function::setVariableType()");
        }
    }

    bool isSubset(const int set1, const int set2) const
    {
        if (set1 == set2) {
            return true;
        }
        auto it = findSet(set1);
        if (it == local_entitysets_.end()) {
            yyerror("internal compiler error in Function::isSubset()");
            return false;
        }
        if (it->subsetIndex() == set2) {
            return true;
        }
        if (it->subsetIndex() == set1) {
            return false;
        }
        return isSubset(it->subsetIndex(), set2);
    }

    std::string name() const
    {
        return name_;
    }

    const FunctionType& functionType() const
    {
        return type_;
    }

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const
    {
        return type_.returnType(argtypes);
    }

private:
    std::pair<bool, EquelleType> declared(const std::string& name) const
    {
        auto lit = local_variables_.find(name);
        if (lit != local_variables_.end()) {
            return std::make_pair(true, lit->type());
        }
        auto ait = std::find_if(type_.arguments().begin(), type_.arguments().end(),
                                [&](const Variable& a) { return a.name() == name; });
        if (ait != type_.arguments().end()) {
            return std::make_pair(true, ait->type());
        }
        return std::make_pair(false, EquelleType());
    }

    std::vector<EntitySet>::const_iterator findSet(const int index) const
    {
        return std::find_if(local_entitysets_.begin(), local_entitysets_.end(),
                            [&](const EntitySet& es) { return es.index() == index; });
    }
    std::string name_;
    std::set<Variable> local_variables_;
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

    static int declareNewEntitySet(const int subset_entity_index)
    {
        return instance().current_function_->declareEntitySet(instance().next_subset_index_++, subset_entity_index);
    }

    static bool isVariableDeclared(const std::string& name)
    {
        return instance().current_function_->isVariableDeclared(name);
    }

    static bool isVariableAssigned(const std::string& name)
    {
        return instance().current_function_->isVariableAssigned(name);
    }

    static void setVariableAssigned(const std::string& name, const bool assigned)
    {
        return instance().current_function_->setVariableAssigned(name, assigned);
    }

    static EquelleType variableType(const std::string& name)
    {
        return instance().current_function_->variableType(name);
    }

    static void setVariableType(const std::string& name, const EquelleType& type)
    {
        return instance().current_function_->setVariableType(name, type);
    }

    static bool isFunctionDeclared(const std::string& name)
    {
        return instance().isFunctionDeclaredImpl(name);
    }

    static const Function& getFunction(const std::string& name)
    {
        return instance().getFunctionImpl(name);
    }

    static const Function& getCurrentFunction()
    {
        return *instance().current_function_;
    }

    /// Returns true if set1 is a (non-strict) subset of set2.
    static bool isSubset(const int set1, const int set2)
    {
        return instance().current_function_->isSubset(set1, set2);
    }

private:
    SymbolTable()
        : next_subset_index_(FirstRuntimeEntitySet)
    {
        // ----- Add built-in functions to function table. -----
        // 1. Grid functions.
        functions_.emplace_back("Main", FunctionType(EquelleType()));
        functions_.emplace_back("InteriorCells", FunctionType(EquelleType(Cell, true, InteriorCells)));
        functions_.emplace_back("BoundaryCells", FunctionType(EquelleType(Cell, true, BoundaryCells)));
        functions_.emplace_back("AllCells", FunctionType(EquelleType(Cell, true, AllCells)));
        functions_.emplace_back("InteriorFaces", FunctionType(EquelleType(Cell, true, InteriorFaces)));
        functions_.emplace_back("BoundaryFaces", FunctionType(EquelleType(Cell, true, BoundaryFaces)));
        functions_.emplace_back("AllFaces", FunctionType(EquelleType(Cell, true, AllFaces)));
        functions_.emplace_back("InteriorEdges", FunctionType(EquelleType(Cell, true, InteriorEdges)));
        functions_.emplace_back("BoundaryEdges", FunctionType(EquelleType(Cell, true, BoundaryEdges)));
        functions_.emplace_back("AllEdges", FunctionType(EquelleType(Cell, true, AllEdges)));
        functions_.emplace_back("InteriorVertices", FunctionType(EquelleType(Cell, true, InteriorVertices)));
        functions_.emplace_back("BoundaryVertices", FunctionType(EquelleType(Cell, true, BoundaryVertices)));
        functions_.emplace_back("AllVertices", FunctionType(EquelleType(Cell, true, AllVertices)));
        functions_.emplace_back("FirstCell",
                                FunctionType({ Variable("faces", EquelleType()) },
                                             EquelleType(Cell, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        functions_.emplace_back("SecondCell",
                                FunctionType({ Variable("faces", EquelleType()) },
                                             EquelleType(Cell, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        functions_.emplace_back("IsEmpty",
                                FunctionType({ Variable("entities", EquelleType()) },
                                             EquelleType(Bool, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        functions_.emplace_back("Centroid",
                                FunctionType({ Variable("entities", EquelleType()) },
                                             EquelleType(Vector, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        functions_.emplace_back("Normal",
                                FunctionType({ Variable("faces", EquelleType()) },
                                             EquelleType(Vector, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        // 2. User input functions.
        functions_.emplace_back("UserSpecifiedScalarWithDefault",
                                FunctionType({ Variable("default", EquelleType(Scalar)) },
                                             EquelleType(Scalar)));
        functions_.emplace_back("UserSpecifiedCollectionOfScalar",
                                FunctionType({ Variable("entities", EquelleType()) },
                                             EquelleType(Scalar, true),
                                             { InvalidIndex, 0, InvalidIndex}));
        functions_.emplace_back("UserSpecifiedCollectionOfFaceSubsetOf",
                                FunctionType({ Variable("entities", EquelleType()) },
                                             EquelleType(Face, true),
                                             { InvalidIndex, InvalidIndex, 0}));
        // 3. Discrete operators.
        functions_.emplace_back("Gradient",
                                FunctionType({ Variable("values", EquelleType(Scalar, true, AllCells)) },
                                             EquelleType(Scalar, true, InteriorFaces)));
        functions_.emplace_back("Divergence",
                                FunctionType({ Variable("values", EquelleType()) },
                                             EquelleType(Scalar, true, AllCells)));
        // 4. Other functions
        functions_.emplace_back("NewtonSolve",
                                FunctionType({ Variable("u", EquelleType()) },
                                             EquelleType(Scalar, true, AllCells)));
        functions_.emplace_back("Output",
                                FunctionType({ Variable("data", EquelleType()) },
                                             EquelleType()));


        // ----- Set main function ref and current (initially equal to main). -----
        main_function_ = functions_.begin();
        current_function_ = main_function_;

        // ----- Add built-in entity sets to entity set table. -----
        main_function_->declareEntitySet(InteriorCells, AllCells);
        main_function_->declareEntitySet(BoundaryCells, AllCells);
        main_function_->declareEntitySet(AllCells, AllCells);
        main_function_->declareEntitySet(InteriorFaces, AllFaces);
        main_function_->declareEntitySet(BoundaryFaces, AllFaces);
        main_function_->declareEntitySet(AllFaces, AllFaces);
        main_function_->declareEntitySet(InteriorEdges, AllEdges);
        main_function_->declareEntitySet(BoundaryEdges, AllEdges);
        main_function_->declareEntitySet(AllEdges, AllEdges);
        main_function_->declareEntitySet(InteriorVertices, AllVertices);
        main_function_->declareEntitySet(BoundaryVertices, AllVertices);
        main_function_->declareEntitySet(AllVertices, AllVertices);
    }

    static SymbolTable& instance()
    {
        static SymbolTable s;
        return s;
    }

    /// Used only for setting up initial built-in entity sets.
    static int declareEntitySet(const int entity_index, const int subset_entity_index)
    {
        return instance().current_function_->declareEntitySet(entity_index, subset_entity_index);
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

    bool isFunctionDeclaredImpl(const std::string& name) const
    {
        return findFunction(name) != functions_.end();
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

    std::list<Function>::const_iterator findFunction(const std::string& name) const
    {
        auto it = std::find_if(functions_.begin(), functions_.end(),
                               [&](const Function& f) { return f.name() == name; });
        return it;
    }

    int next_subset_index_;
    std::list<Function> functions_;
    std::list<Function>::iterator main_function_;
    std::list<Function>::iterator current_function_;
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

class SequenceNode : public Node
{
public:
    void pushNode(Node* node)
    {
        nodes_.push_back(node);
    }
private:
    std::vector<Node*> nodes_;
};

class NumberNode : public Node
{
public:
    NumberNode(const double num) : num_(num) {}
    EquelleType type() const
    {
        return EquelleType(Scalar);
    }
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
    EquelleType type() const
    {
        EquelleType lt = left_->type();
        EquelleType rt = right_->type();
        switch (op_) {
        case Add:
            return lt; // should be identical to rt.
        case Subtract:
            return lt; // should be identical to rt.
        case Multiply: {
            const bool isvec = lt.basicType() == Vector || rt.basicType() == Vector; 
            const BasicType bt = isvec ? Vector : Scalar;
            const bool coll = lt.isCollection() || rt.isCollection();
            const int gm = lt.isCollection() ? lt.gridMapping() : rt.gridMapping();
            return EquelleType(bt, coll, gm);
        }
        case Divide: {
            const BasicType bt = lt.basicType();
            const bool coll = lt.isCollection() || rt.isCollection();
            const int gm = lt.isCollection() ? lt.gridMapping() : rt.gridMapping();
            return EquelleType(bt, coll, gm);
        }
        default:
            yyerror("internal compiler error in BinaryOpNode::type().");
            return EquelleType();
        }
    }
private:
    BinaryOp op_;
    NodePtr left_;
    NodePtr right_;
};

class NormNode : public Node
{
public:
    NormNode(NodePtr expr_to_norm) : expr_to_norm_(expr_to_norm){}
    EquelleType type() const
    {
        return EquelleType(Scalar,
                           expr_to_norm_->type().isCollection(),
                           expr_to_norm_->type().gridMapping());
    }
private:
    NodePtr expr_to_norm_;
};

class UnaryNegationNode : public Node
{
public:
    UnaryNegationNode(NodePtr expr_to_negate) : expr_to_negate_(expr_to_negate) {}
    EquelleType type() const
    {
        return expr_to_negate_->type();
    }
private:
    NodePtr expr_to_negate_;
};

class OnNode : public Node
{
public:
    OnNode(NodePtr left, NodePtr right) : left_(left), right_(right) {}
    EquelleType type() const
    {
        return EquelleType(left_->type().basicType(), true, right_->type().gridMapping());
    }
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
    std::vector<Variable> arguments() const
    {
        std::vector<Variable> args;
        args.reserve(decls_.size());
        for (auto vdn : decls_) {
            args.push_back(Variable(vdn->name(), vdn->type(), true));
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

class FuncArgsNode : public Node
{
public:
    FuncArgsNode(Node* expr = 0)
    {
        if (expr) {
            args_.push_back(expr);
        }
    }
    void addArg(Node* expr)
    {
        args_.push_back(expr);
    }
    std::vector<EquelleType> argumentTypes() const
    {
        std::vector<EquelleType> argtypes;
        argtypes.reserve(args_.size());
        for (auto np : args_) {
            argtypes.push_back(np->type());
        }
        return argtypes;
    }
private:
    std::vector<Node*> args_;
};

class FuncCallNode : public Node
{
public:
    FuncCallNode(const std::string& funcname,
                 FuncArgsNode* funcargs,
                 const int dynamic_subset_return = NotApplicable)
        : funcname_(funcname), funcargs_(funcargs), dsr_(dynamic_subset_return)
    {}
    EquelleType type() const
    {
        EquelleType t = SymbolTable::getFunction(funcname_).returnType(funcargs_->argumentTypes());
        if (dsr_ != NotApplicable) {
            assert(t.isEntityCollection());
            return EquelleType(t.basicType(), true, dsr_);
        } else {
            return t;
        }
    }
private:
    std::string funcname_;
    FuncArgsNode* funcargs_;
    int dsr_;
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

inline VarAssignNode* handleAssignment(const std::string name, NodePtr expr)
{
    // If already declared...
    if (SymbolTable::isVariableDeclared(name)) {
        // Check if already assigned.
        if (SymbolTable::isVariableAssigned(name)) {
            std::string err_msg = "variable already assigned, cannot re-assign ";
            err_msg += name;
            yyerror(err_msg.c_str());
            return nullptr;
        }
        // Check that declared type matches right hand side.
        EquelleType lhs_type = SymbolTable::variableType(name);
        EquelleType rhs_type = expr->type();
        if (lhs_type != rhs_type) {
            // Check for special case: variable declared to have type
            // 'Collection Of <Entity> Subset Of <Some entityset>',
            // actual setting its grid mapping is postponed until the entityset
            // has been created by a function call.
            // That means we have to set the actual grid mapping here.
            if (lhs_type.gridMapping() == PostponedDefinition
                && lhs_type.basicType() == rhs_type.basicType()
                && lhs_type.isCollection() && rhs_type.isCollection()
                && SymbolTable::isSubset(rhs_type.gridMapping(), lhs_type.subsetOf())) {
                // OK, should make postponed definition of the variable.
                SymbolTable::setVariableType(name, rhs_type);
            } else {
                std::string err_msg = "mismatch between type in assignment and declaration for ";
                err_msg += name;
                yyerror(err_msg.c_str());
                return nullptr;
            }
        }
    } else {
        SymbolTable::declareVariable(name, expr->type());
    }

    // Set variable to assigned and return.
    SymbolTable::setVariableAssigned(name, true);
    return new VarAssignNode(name, expr);
}

inline NodePtr handleFuncDeclaration(const std::string name, FuncTypeNodePtr ftype)
{
    SymbolTable::declareFunction(name, ftype->funcType());
    return new FuncDeclNode(name, ftype);
}

inline NodePtr handleDeclarationAssign(const std::string name, TypeNodePtr type, NodePtr expr)
{
    SequenceNode* seq = new SequenceNode;
    seq->pushNode(handleDeclaration(name, type));
    seq->pushNode(handleAssignment(name, expr));
    return seq;
}

inline TypeNodePtr handleCollection(TypeNodePtr btype, NodePtr gridmapping, NodePtr subsetof)
{
    assert(gridmapping == nullptr || subsetof == nullptr);
    EquelleType bt = btype->type();
    if (!bt.isBasic()) {
        std::string errmsg = "attempting to declare a Collection Of <nonsense>";
        yyerror(errmsg.c_str());
    }
    int gm = NotApplicable;
    if (gridmapping) {
        if (!gridmapping->type().isEntityCollection() || gridmapping->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be On a Collection of Cell, Face etc.");
        } else {
            gm = gridmapping->type().gridMapping();
        }
    }
    int subset = NotApplicable;
    if (subsetof) {
        // We are creating a new entity collection.
        if (!subsetof->type().isEntityCollection() || subsetof->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be Subset Of a Collection of Cell, Face etc.");
        } else {
            gm = PostponedDefinition;
            subset = subsetof->type().gridMapping();
        }
    }
    return new TypeNode(EquelleType(bt.basicType(), true, gm, subset));
}

inline FuncTypeNodePtr handleFuncType(FuncArgsDeclNode* argtypes, TypeNodePtr rtype)
{
    return new FuncTypeNode(FunctionType(argtypes->arguments(), rtype->type()));
}

inline FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args)
{
    const Function& f = SymbolTable::getFunction(name);
    int dynsubret = f.functionType().dynamicSubsetReturn(args->argumentTypes());
    if (dynsubret != NotApplicable) {
        // Create a new entity collection. This is the only place this can happen.
        const int gm = SymbolTable::declareNewEntitySet(dynsubret);
        return new FuncCallNode(name, args, gm);
    } else {
        return new FuncCallNode(name, args);
    }
}

inline BinaryOpNode* handleBinaryOp(BinaryOp op, Node* left, Node* right)
{
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if (!isNumericType(lt.basicType()) || !(isNumericType(rt.basicType()))) {
        yyerror("arithmetic binary operators only apply to numeric types");
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            yyerror("arithmetic binary operators on 'Collection's only acceptable "
                    "if both sides are 'On' the same set.");
        }
    }
    switch (op) {
    case Add:
        // Intentional fall-through.
    case Subtract:
        if (lt != rt) {
            yyerror("addition and subtraction only allowed between identical types.");
        }
        break;
    case Multiply:
        if (lt.basicType() == Vector && rt.basicType() == Vector) {
            yyerror("cannot multiply two 'Vector' types.");
        }
        break;
    case Divide:
        if (rt.basicType() != Scalar) {
            yyerror("can only divide by 'Scalar' types");
        }
        break;
    default:
        yyerror("internal compiler error in handleBinaryOp().");
    }
    return new BinaryOpNode(op, left, right);
}

inline NormNode* handleNorm(NodePtr expr_to_norm)
{
    const BasicType bt = expr_to_norm->type().basicType();
    if (isEntityType(bt) || bt == Scalar || bt == Vector) {
        return new NormNode(expr_to_norm);
    } else {
        yyerror("can only take norm of Scalar, Vector, Cell, Face, Edge and Vertex types.");
        return new NormNode(expr_to_norm);
    }
}

#endif // PARSER_HEADER_INCLUDED
