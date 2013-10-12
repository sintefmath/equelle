/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef ASTNODES_HEADER_INCLUDED
#define ASTNODES_HEADER_INCLUDED

#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"
#include "ASTVisitorInterface.hpp"

#include <vector>
#include <cassert>

// ------ Abstract syntax tree classes ------


/// Base class for all AST classes.
class Node
{
public:
    virtual ~Node()
    {}
    virtual EquelleType type() const
    {
        return EquelleType();
    }
    virtual void accept(ASTVisitorInterface& visitor) = 0;
};




class SequenceNode : public Node
{
public:
    void pushNode(Node* node)
    {
        if (node) {
            nodes_.push_back(node);
        }
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);

        const size_t n = nodes_.size();
        for (size_t i = 0; i < n; ++i) {
            nodes_[i]->accept(visitor);
            if (i < n - 1) {
                visitor.midVisit(*this);
            }
        }
        visitor.postVisit(*this);
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    double number() const
    {
        return num_;
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    EquelleType et_;
};




class FuncTypeNode : public Node
{
public:
    FuncTypeNode(const FunctionType ft) : ft_(ft) {}
    FunctionType funcType() const
    {
        return ft_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    FunctionType ft_;
};




enum BinaryOp { Add, Subtract, Multiply, Divide };


class BinaryOpNode : public Node
{
public:
    BinaryOpNode(BinaryOp op, Node* left, Node* right) : op_(op), left_(left), right_(right) {}
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
    BinaryOp op() const
    {
        return op_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        left_->accept(visitor);
        visitor.midVisit(*this);
        right_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    BinaryOp op_;
    Node* left_;
    Node* right_;
};




class NormNode : public Node
{
public:
    NormNode(Node* expr_to_norm) : expr_to_norm_(expr_to_norm){}
    EquelleType type() const
    {
        return EquelleType(Scalar,
                           expr_to_norm_->type().isCollection(),
                           expr_to_norm_->type().gridMapping());
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_to_norm_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* expr_to_norm_;
};




class UnaryNegationNode : public Node
{
public:
    UnaryNegationNode(Node* expr_to_negate) : expr_to_negate_(expr_to_negate) {}
    EquelleType type() const
    {
        return expr_to_negate_->type();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_to_negate_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* expr_to_negate_;
};




class OnNode : public Node
{
public:
    OnNode(Node* left, Node* right) : left_(left), right_(right) {}
    EquelleType type() const
    {
        return EquelleType(left_->type().basicType(), true, right_->type().gridMapping());
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        left_->accept(visitor);
        visitor.midVisit(*this);
        right_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* left_;
    Node* right_;
};




class TrinaryIfNode : public Node
{
public:
    TrinaryIfNode(Node* predicate, Node* iftrue, Node* iffalse)
        : predicate_(predicate), iftrue_(iftrue), iffalse_(iffalse)
    {}
    EquelleType type() const
    {
        return iftrue_->type();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        predicate_->accept(visitor);
        visitor.questionMarkVisit(*this);
        iftrue_->accept(visitor);
        visitor.colonVisit(*this);
        iffalse_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* predicate_;
    Node* iftrue_;
    Node* iffalse_;
};




class VarDeclNode : public Node
{
public:
    VarDeclNode(std::string varname, TypeNode* type)
        : varname_(varname), type_(type)
    {
    }
    EquelleType type() const
    {
        return type_->type();
    }
    const std::string& name() const
    {
        return varname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        type_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string varname_;
    TypeNode* type_;
};




class VarAssignNode : public Node
{
public:
    VarAssignNode(std::string varname, Node* expr) : varname_(varname), expr_(expr)
    {
    }
    const std::string& name() const
    {
        return varname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string varname_;
    Node* expr_;
};




class VarNode : public Node
{
public:
    VarNode(const std::string& varname) : varname_(varname)
    {
    }
    EquelleType type() const
    {
        return SymbolTable::variableType(varname_);
    }
    const std::string& name() const
    {
        return varname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    std::string varname_;
};




class FuncRefNode : public Node
{
public:
    FuncRefNode(const std::string& funcname) : funcname_(funcname)
    {
    }
    EquelleType type() const
    {
        // Functions' types cannot be expressed as an EquelleType
        return EquelleType();
    }
    const std::string& name() const
    {
        return funcname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    std::string funcname_;
};




class JustAnIdentifierNode : public Node
{
public:
    JustAnIdentifierNode(const std::string& id) : id_(id)
    {
    }
    const std::string& name() const
    {
        return id_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    std::string id_;
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        const size_t n = decls_.size();
        for (size_t i = 0; i < n; ++i) {
            decls_[i]->accept(visitor);
            if (i < n - 1) {
                visitor.midVisit(*this);
            }
        }
        visitor.postVisit(*this);
    }
private:
    std::vector<VarDeclNode*> decls_;
};




class FuncDeclNode : public Node
{
public:
    FuncDeclNode(std::string funcname, FuncTypeNode* ftype)
        : funcname_(funcname), ftype_(ftype)
    {
    }
    const std::string& name() const
    {
        return funcname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        ftype_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string funcname_;
    FuncTypeNode* ftype_;
};




class FuncStartNode : public Node
{
public:
    FuncStartNode(std::string funcname, Node* funcargs)
        : funcname_(funcname), funcargs_(funcargs)
    {
    }
    const std::string& name() const
    {
        return funcname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        funcargs_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string funcname_;
    Node* funcargs_;
};



class FuncAssignNode : public Node
{
public:
    FuncAssignNode(Node* funcstart, Node* funcbody)
        : funcstart_(funcstart), funcbody_(funcbody)
    {
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        funcstart_->accept(visitor);
        funcbody_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* funcstart_;
    Node* funcbody_;
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        const size_t n = args_.size();
        for (size_t i = 0; i < n; ++i) {
            args_[i]->accept(visitor);
            if (i < n - 1) {
                visitor.midVisit(*this);
            }
        }
        visitor.postVisit(*this);
    }
private:
    std::vector<Node*> args_;
};




class ReturnStatementNode : public Node
{
public:
    ReturnStatementNode(Node* expr)
        : expr_(expr)
    {}
    EquelleType type() const
    {
        return expr_->type();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* expr_;
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
    const std::string& name() const
    {
        return funcname_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        funcargs_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string funcname_;
    FuncArgsNode* funcargs_;
    int dsr_;
};



class FuncCallStatementNode : public Node
{
public:
    FuncCallStatementNode(FuncCallNode* fcall)
    : fcall_(fcall)
    {}
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        fcall_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    FuncCallNode* fcall_;
};


#endif // ASTNODES_HEADER_INCLUDED
