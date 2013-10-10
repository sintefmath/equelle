/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSER_HEADER_INCLUDED
#define PARSER_HEADER_INCLUDED

#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"

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
