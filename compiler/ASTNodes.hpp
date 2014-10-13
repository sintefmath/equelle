/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef ASTNODES_HEADER_INCLUDED
#define ASTNODES_HEADER_INCLUDED

#include "NodeInterface.hpp"
#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"
#include "ASTVisitorInterface.hpp"
#include "Dimension.hpp"
#include "EquelleUnits.hpp"

#include <vector>
#include <cassert>



// ------ Abstract syntax tree classes ------



/// Base class for expression nodes.
class ExpressionNode : public Node
{
public:
    virtual EquelleType type() const = 0;
    virtual Dimension dimension() const
    {
        Dimension d;
        // To make errors stand out.
        d.setCoefficient(LuminousIntensity, -999);
        return d;
    }
};


class SequenceNode : public Node
{
public:
    virtual ~SequenceNode()
    {
        for (auto np : nodes_) {
            delete np;
        }
    }
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
    const std::vector<Node*>& nodes() {
        return nodes_;
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




class StringNode : public ExpressionNode
{
public:
    StringNode(const std::string& content) : content_(content) {}
    EquelleType type() const
    {
        return EquelleType(String);
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    const std::string& content() const
    {
        return content_;
    }
private:
    std::string content_;
};




class TypeNode : public Node
{
public:
    TypeNode(const EquelleType et) : et_(et) {}
    virtual EquelleType type() const
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




class CollectionTypeNode : public TypeNode
{
public:
    CollectionTypeNode(TypeNode* btype, ExpressionNode* gridmapping, ExpressionNode* subsetof)
        : TypeNode(EquelleType()),
          btype_(btype),
          gridmapping_(gridmapping),
          subsetof_(subsetof)
    {
        // TODO. Check if this assert is what we want.
        //       Is "Collection Of X On Y Subset Of Z" allowed?
        assert(gridmapping == nullptr || subsetof == nullptr);
    }
    ~CollectionTypeNode()
    {
        delete btype_;
        delete gridmapping_;
        delete subsetof_;
    }
    const TypeNode* baseType() const
    {
        return btype_;
    }
    const ExpressionNode* gridMapping() const
    {
        return gridmapping_;
    }
    const ExpressionNode* subsetOf() const
    {
        return subsetof_;
    }
    EquelleType type() const
    {
        EquelleType bt = btype_->type();
        int gm = NotApplicable;
        if (gridmapping_) {
            gm = gridmapping_->type().gridMapping();
        }
        int subset = NotApplicable;
        if (subsetof_) {
            gm = PostponedDefinition;
            subset = subsetof_->type().gridMapping();
        }
        return EquelleType(bt.basicType(), Collection, gm, subset);
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        // btype_->accept(visitor);
        if (gridmapping_) {
            gridmapping_->accept(visitor);
        }
        if (subsetof_) {
            subsetof_->accept(visitor);
        }
        visitor.postVisit(*this);
    }

private:
    TypeNode* btype_;
    ExpressionNode* gridmapping_;
    ExpressionNode* subsetof_;
};




enum BinaryOp { Add, Subtract, Multiply, Divide };


class BinaryOpNode : public ExpressionNode
{
public:
    BinaryOpNode(BinaryOp op, ExpressionNode* left, ExpressionNode* right)
        : op_(op), left_(left), right_(right)
    {
    }
    virtual ~BinaryOpNode()
    {
        delete left_;
        delete right_;
    }
    EquelleType type() const
    {
        EquelleType lt = left_->type();
        EquelleType rt = right_->type();
        if (lt.isSequence() || rt.isSequence()) {
            yyerror("internal compiler error in BinaryOpNode::type(), sequences not allowed");
            return EquelleType();
        }
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
            return EquelleType(bt, coll ? Collection : None, gm);
        }
        case Divide: {
            const BasicType bt = lt.basicType();
            const bool coll = lt.isCollection() || rt.isCollection();
            const int gm = lt.isCollection() ? lt.gridMapping() : rt.gridMapping();
            return EquelleType(bt, coll ? Collection : None, gm);
        }
        default:
            yyerror("internal compiler error in BinaryOpNode::type().");
            return EquelleType();
        }
    }
    Dimension dimension() const
    {
        switch (op_) {
        case Add:
            return left_->dimension(); // Should be identical to right->dimension().
        case Subtract:
            return left_->dimension(); // Should be identical to right->dimension().
        case Multiply:
            return left_->dimension() + right_->dimension();
        case Divide:
            return left_->dimension() - right_->dimension();
        default:
            yyerror("internal compiler error in BinaryOpNode::type().");
            return Dimension();
        }
    }
    BinaryOp op() const
    {
        return op_;
    }
    const ExpressionNode* left() const
    {
        return left_;
    }
    const ExpressionNode* right() const
    {
        return right_;
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
    ExpressionNode* left_;
    ExpressionNode* right_;
};




enum ComparisonOp { Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual };


class ComparisonOpNode : public ExpressionNode
{
public:
    ComparisonOpNode(ComparisonOp op, ExpressionNode* left, ExpressionNode* right)
        : op_(op), left_(left), right_(right)
    {
    }
    virtual ~ComparisonOpNode()
    {
        delete left_;
        delete right_;
    }
    EquelleType type() const
    {
        EquelleType lt = left_->type();
        return EquelleType(Bool, lt.compositeType(), lt.gridMapping());
    }
    ComparisonOp op() const
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
    ComparisonOp op_;
    ExpressionNode* left_;
    ExpressionNode* right_;
};




class NormNode : public ExpressionNode
{
public:
    NormNode(ExpressionNode* expr_to_norm) : expr_to_norm_(expr_to_norm){}
    virtual ~NormNode()
    {
        delete expr_to_norm_;
    }
    EquelleType type() const
    {
        return EquelleType(Scalar,
                           expr_to_norm_->type().compositeType(),
                           expr_to_norm_->type().gridMapping());
    }
    Dimension dimension() const
    {
        EquelleType t = expr_to_norm_->type();
        if (isNumericType(t.basicType())) {
            // The norm of a Scalar or Vector has the same dimension
            // as the Scalar or Vector itself.
            return expr_to_norm_->dimension();
        } else {
            // Taking the norm of a grid entity.
            // Note: for now we always assume 3d for the
            // purpose of dimensions of these types.
            Dimension d;
            switch (t.basicType()) {
            case Vertex:
                // 0-dimensional.
                break;
            case Edge:
                d.setCoefficient(Length, 1);
                break;
            case Face:
                d.setCoefficient(Length, 2);
                break;
            case Cell:
                d.setCoefficient(Length, 3);
                break;
            default:
                yyerror("internal compiler error in NormNode::dimension().");
            }
            // TODO: we use a dimensionless value here now, until
            // dimension of function calls has been implemented.
            d = Dimension();
            return d;
        }
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_to_norm_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    ExpressionNode* expr_to_norm_;
};




class UnaryNegationNode : public ExpressionNode
{
public:
    UnaryNegationNode(ExpressionNode* expr_to_negate) : expr_to_negate_(expr_to_negate) {}
    virtual ~UnaryNegationNode()
    {
        delete expr_to_negate_;
    }
    EquelleType type() const
    {
        return expr_to_negate_->type();
    }
    Dimension dimension() const
    {
        return expr_to_negate_->dimension();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_to_negate_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    ExpressionNode* expr_to_negate_;
};




class OnNode : public ExpressionNode
{
public:
    OnNode(ExpressionNode* left, ExpressionNode* right, bool is_extend)
        : left_(left), right_(right), is_extend_(is_extend)
    {
    }
    virtual ~OnNode()
    {
        delete left_;
        delete right_;
    }
    EquelleType type() const
    {
        return EquelleType(left_->type().basicType(), Collection, right_->type().gridMapping(), left_->type().subsetOf());
    }
    EquelleType lefttype() const
    {
        return left_->type();
    }
    bool isExtend() const
    {
        return is_extend_;
    }
    Dimension dimension() const
    {
        return left_->dimension();
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
    ExpressionNode* left_;
    ExpressionNode* right_;
    bool is_extend_;
};




class TrinaryIfNode : public ExpressionNode
{
public:
    TrinaryIfNode(ExpressionNode* predicate, ExpressionNode* iftrue, ExpressionNode* iffalse)
        : predicate_(predicate), iftrue_(iftrue), iffalse_(iffalse)
    {}
    virtual ~TrinaryIfNode()
    {
        delete predicate_;
        delete iftrue_;
        delete iffalse_;
    }
    EquelleType type() const
    {
        return iftrue_->type();
    }
    Dimension dimension() const
    {
        return iftrue_->dimension(); // Should be identical to iffalse_->dimension().
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
    ExpressionNode* predicate_;
    ExpressionNode* iftrue_;
    ExpressionNode* iffalse_;
};




class VarDeclNode : public Node
{
public:
    VarDeclNode(std::string varname, TypeNode* type)
        : varname_(varname), type_(type)
    {
    }
    virtual ~VarDeclNode()
    {
        delete type_;
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
    VarAssignNode(std::string varname, ExpressionNode* expr) : varname_(varname), expr_(expr)
    {
    }
    virtual ~VarAssignNode()
    {
        delete expr_;
    }
    const std::string& name() const
    {
        return varname_;
    }
    EquelleType type() const
    {
        return expr_->type();
    }
    const ExpressionNode* rhs() const
    {
        return expr_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string varname_;
    ExpressionNode* expr_;
};




class VarNode : public ExpressionNode
{
public:
    VarNode(const std::string& varname) : varname_(varname)
    {
    }
    EquelleType type() const
    {
        // We do not want mutability of a variable to be passed on to
        // expressions involving that variable.
        if (SymbolTable::isVariableDeclared(varname_)) {
            EquelleType et = SymbolTable::variableType(varname_);
            if (et.isMutable()) {
                et.setMutable(false);
            }
            return et;
        } else if (SymbolTable::isFunctionDeclared(varname_)) {
            // Function reference.
            return EquelleType();
        } else {
            throw std::logic_error("Internal compiler error in VarNode::type().");
        }
    }
    Dimension dimension() const
    {
        return SymbolTable::variableDimension(varname_);
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




class FuncArgsDeclNode : public Node
{
public:
    FuncArgsDeclNode(VarDeclNode* vardecl = 0)
    {
        if (vardecl) {
            decls_.push_back(vardecl);
        }
    }
    virtual ~FuncArgsDeclNode()
    {
        for (auto decl : decls_) {
            delete decl;
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




class FuncTypeNode : public Node
{
public:
    FuncTypeNode(FuncArgsDeclNode* argtypes, TypeNode* rtype)
        : argtypes_(argtypes), rtype_(rtype)
    {
    }
    FunctionType funcType() const
    {
        return FunctionType(argtypes_->arguments(), rtype_->type());
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    FuncArgsDeclNode* argtypes_;
    TypeNode* rtype_;
};



class FuncRefNode : public ExpressionNode
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




class JustAnIdentifierNode : public ExpressionNode
{
public:
    JustAnIdentifierNode(const std::string& id) : id_(id)
    {
    }
    EquelleType type() const
    {
        return EquelleType();
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




class FuncDeclNode : public Node
{
public:
    FuncDeclNode(std::string funcname, FuncTypeNode* ftype)
        : funcname_(funcname), ftype_(ftype)
    {
    }
    virtual ~FuncDeclNode()
    {
        delete ftype_;
    }
    const std::string& name() const
    {
        return funcname_;
    }
    const FuncTypeNode* ftype() const
    {
        return ftype_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        ftype_->accept(visitor);
        visitor.postVisit(*this);
#if 0
        SymbolTable::setCurrentFunction(funcname_);
        visitor.visit(*this);
        ftype_->accept(visitor);
        visitor.postVisit(*this);
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
#endif
    }
private:
    std::string funcname_;
    FuncTypeNode* ftype_;
};





class FuncArgsNode : public Node
{
public:
    FuncArgsNode(ExpressionNode* expr = 0)
    {
        if (expr) {
            args_.push_back(expr);
        }
    }
    virtual ~FuncArgsNode()
    {
        for (auto arg : args_) {
            delete arg;
        }
    }
    void addArg(ExpressionNode* expr)
    {
        args_.push_back(expr);
    }
    const std::vector<ExpressionNode*>& arguments() const
    {
        return args_;
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
    std::vector<ExpressionNode*> args_;
};




class ReturnStatementNode : public Node
{
public:
    ReturnStatementNode(ExpressionNode* expr)
    : expr_(expr)
    {}
    virtual ~ReturnStatementNode()
    {
        delete expr_;
    }
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
    ExpressionNode* expr_;
};


/**
 * This class handles all types of statements that look like function calls.
 * Right now, this might be
 * a stencil access: u(i, j),
 * or a function call: computeResidual(u)
 * In addition, it might be a function definition, or stencil assignment:
 * u(i, j) = 5.0
 * computeResidual(u) = { ... }
 */
class FuncCallLikeNode : public ExpressionNode
{
public:
    virtual const std::string& name() const = 0;
    virtual const FuncArgsNode* args() const = 0;
};

class FuncStartNode : public FuncCallLikeNode
{
public:
    FuncStartNode(std::string funcname, FuncArgsNode* funcargs)
        : funcname_(funcname), funcargs_(funcargs)
    {
    }
    virtual ~FuncStartNode()
    {
        delete funcargs_;
    }
    const std::string& name() const
    {
        return funcname_;
    }
    virtual const FuncArgsNode* args() const
    {
        return funcargs_;
    }
    EquelleType type() const
    {
        return EquelleType();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
#if 0
        SymbolTable::setCurrentFunction(funcname_);
#endif
        visitor.visit(*this);
        funcargs_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string funcname_;
    FuncArgsNode* funcargs_;
};



class FuncAssignNode : public Node
{
public:
    FuncAssignNode(FuncStartNode* funcstart, Node* funcbody)
        : funcstart_(funcstart), funcbody_(funcbody)
    {
    }
    virtual ~FuncAssignNode()
    {
        delete funcstart_;
        delete funcbody_;
    }
    const std::string& name() const
    {
        return funcstart_->name();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        funcstart_->accept(visitor);
        funcbody_->accept(visitor);
        visitor.postVisit(*this);
#if 0
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
#endif
    }
private:
    FuncStartNode* funcstart_;
    Node* funcbody_;
};




class StencilNode : public FuncCallLikeNode
{
public:
    StencilNode(const std::string& varname,
                FuncArgsNode* args)
        : varname_(varname), args_(args)
    {}

    virtual ~StencilNode()
    {
        delete args_;
    }

    EquelleType type() const
    {
        // All stencils are at this time scalars
        // We do not want mutability of a variable to be passed on to
        // expressions involving that variable.
        EquelleType et = SymbolTable::variableType(varname_);
        if (et.isMutable()) {
            et.setMutable(false);
        }
        return et;
    }

    virtual const std::string& name() const
    {
        return varname_;
    }

    virtual const FuncArgsNode* args() const
    {
        return args_;
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        args_->accept(visitor);
        visitor.postVisit(*this);
    }

private:
    std::string varname_;
    FuncArgsNode* args_;
};

class FuncCallNode : public FuncCallLikeNode
{
public:
    FuncCallNode(const std::string& funcname,
                 FuncArgsNode* funcargs,
                 const int dynamic_subset_return = NotApplicable)
        : funcname_(funcname), funcargs_(funcargs),
          dsr_(dynamic_subset_return)
    {}

    virtual ~FuncCallNode()
    {
        delete funcargs_;
    }

    void setDynamicSubsetReturn(const int dynamic_subset_return)
    {
        dsr_ = dynamic_subset_return;
    }

    EquelleType type() const
    {
        EquelleType t = SymbolTable::getFunction(funcname_).returnType(funcargs_->argumentTypes());
        if (dsr_ != NotApplicable) {
            assert(t.isEntityCollection());
            return EquelleType(t.basicType(), Collection, dsr_, t.subsetOf(), t.isMutable(), t.isDomain());
        } else {
            return t;
        }
    }

    Dimension dimension() const
    {
        // yyerror("FuncCallNode()::dimension() not implemented");
        // return ExpressionNode::dimension();
        // TODO: function calls are dimensionless for now, fix.
        return Dimension();
    }

    const std::string& name() const
    {
        return funcname_;
    }

    virtual const FuncArgsNode* args() const
    {
        return funcargs_;
    }
    FuncArgsNode& argumentsNode() const
    {
        return *funcargs_;
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
    FuncCallStatementNode(FuncCallNode* func_call)
    : func_call_(func_call)
    {}

    virtual ~FuncCallStatementNode()
    {
        delete func_call_;
    }

    EquelleType type() const
    {
        return func_call_->type();
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        func_call_->accept(visitor);
        visitor.postVisit(*this);
    }

private:
    FuncCallNode* func_call_;
};



class LoopNode : public Node
{
public:
    LoopNode(const std::string& loop_variable,
             const std::string& loop_set,
             SequenceNode* loop_block = 0)
        : loop_variable_(loop_variable),
          loop_set_(loop_set),
          loop_block_(loop_block)
    {
    }
    virtual ~LoopNode()
    {
        delete loop_block_;
    }
    const std::string& loopVariable() const
    {
        return loop_variable_;
    }
    const std::string& loopSet() const
    {
        return loop_set_;
    }
    const std::string& loopName() const
    {
        return loop_name_;
    }
    void setName(const std::string& loop_name)
    {
        loop_name_ = loop_name;
    }
    void setBlock(SequenceNode* loop_block)
    {
        loop_block_ = loop_block;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        loop_block_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string loop_variable_;
    std::string loop_set_;
    std::string loop_name_;
    SequenceNode* loop_block_;
};



class ArrayNode : public ExpressionNode
{
public:
    ArrayNode(FuncArgsNode* expr_list)
    : expr_list_(expr_list)
    {
    }
    virtual ~ArrayNode()
    {
        delete expr_list_;
    }
    EquelleType type() const
    {
        EquelleType t = expr_list_->arguments().front()->type();
        t.setArraySize(expr_list_->arguments().size());
        return t;
    }
    Dimension dimension() const
    {
        // yyerror("ArrayNode()::dimension() not implemented");
        // return ExpressionNode::dimension();
        // TODO: array creation ([x,y,...]) is dimensionless for now, fix.
        return Dimension();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_list_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    FuncArgsNode* expr_list_;
};



class RandomAccessNode : public ExpressionNode
{
public:
    RandomAccessNode(ExpressionNode* expr, const int index)
        : expr_(expr), index_(index)
    {
    }
    virtual ~RandomAccessNode()
    {
        delete expr_;
    }
    int index() const
    {
        return index_;
    }
    bool arrayAccess() const
    {
        return expr_->type().isArray();
    }
    EquelleType type() const
    {
        // Either erpr_ must be an Array, or, if not,
        // we must be a (Collection Of) Scalar,
        // since expr_ must be a (Collection Of) Vector.
        EquelleType t = expr_->type();
        if (t.isArray()) {
            t.setArraySize(NotAnArray);
            return t;
        } else {
            assert(t.basicType() == Vector);
            t.setBasicType(Scalar);
            return t;
        }
    }
    Dimension dimension() const
    {
        // yyerror("RandomAccessNode()::dimension() not implemented");
        // return ExpressionNode::dimension();
        // TODO: array access (a[n]) is dimensionless for now, fix.
        return Dimension();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    ExpressionNode* expr_;
    int index_;
};

class StencilAssignmentNode : public Node
{
public:
    StencilAssignmentNode(StencilNode* lhs, ExpressionNode* rhs)
        : lhs_(lhs), rhs_(rhs)
    {}

    virtual ~StencilAssignmentNode() {
        delete lhs_;
        delete rhs_;
    }

    EquelleType type() const
    {
        return rhs_->type();
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        lhs_->accept(visitor);
        visitor.midVisit(*this);
        rhs_->accept(visitor);
        visitor.postVisit(*this);
    }

    const std::string& name() const {
        return lhs_->name();
    }
private:
    StencilNode* lhs_;
    ExpressionNode* rhs_;
};

class UnitNode : public Node
{
public:
    UnitNode(const std::string& name)
        : conv_factor_(-1e100)
    {
        UnitData ud = unitFromString(name);
        if (ud.valid) {
            dimension_ = ud.dimension;
            conv_factor_ = ud.conv_factor;
        } else {
            yyerror("Unit name not recognised.");
        }
    }

    UnitNode(const Dimension dimension_arg,
             const double conversion_factor_SI)
        : dimension_(dimension_arg),
          conv_factor_(conversion_factor_SI)
    {
    }

    Dimension dimension() const
    {
        return dimension_;
    }

    // The number you must multiply a quantity given in the
    // current unit with to obtain an SI quantity.
    // For example for Inch, the factor ie 0.0254.
    double conversionFactorSI() const
    {
        return conv_factor_;
    }

    EquelleType type() const
    {
        return EquelleType();
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }

private:
    Dimension dimension_;
    double conv_factor_;
};

class QuantityNode : public ExpressionNode
{
public:
    QuantityNode(NumberNode* number_arg, UnitNode* unit_arg)
        : number_(number_arg),
          unit_(unit_arg)
    {
    }

    ~QuantityNode()
    {
        delete number_;
        delete unit_;
    }

    EquelleType type() const
    {
        return EquelleType(Scalar);
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        number_->accept(visitor);
        if (unit_) {
            unit_->accept(visitor);
        }
        visitor.postVisit(*this);
    }

    Dimension dimension() const
    {
        return unit_ ? unit_->dimension() : Dimension();
    }

    double conversionFactorSI() const
    {
        return unit_ ? unit_->conversionFactorSI() : 1.0;
    }

    double number() const
    {
        return number_->number();
    }

private:
    NumberNode* number_;
    UnitNode* unit_;
};


#endif // ASTNODES_HEADER_INCLUDED
