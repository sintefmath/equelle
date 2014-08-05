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

#include <vector>
#include <cassert>

// ------ Abstract syntax tree classes ------


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




class StringNode : public Node
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
    BinaryOpNode(BinaryOp op, Node* left, Node* right)
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




enum ComparisonOp { Less, Greater, LessEqual, GreaterEqual, Equal, NotEqual };


class ComparisonOpNode : public Node
{
public:
    ComparisonOpNode(ComparisonOp op, Node* left, Node* right)
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
    Node* left_;
    Node* right_;
};




class NormNode : public Node
{
public:
    NormNode(Node* expr_to_norm) : expr_to_norm_(expr_to_norm){}
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
    virtual ~UnaryNegationNode()
    {
        delete expr_to_negate_;
    }
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
    OnNode(Node* left, Node* right, bool is_extend)
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
    bool is_extend_;
};




class TrinaryIfNode : public Node
{
public:
    TrinaryIfNode(Node* predicate, Node* iftrue, Node* iffalse)
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
    VarAssignNode(std::string varname, Node* expr) : varname_(varname), expr_(expr)
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
        // We do not want mutability of a variable to be passed on to
        // expressions involving that variable.
        EquelleType et = SymbolTable::variableType(varname_);
        if (et.isMutable()) {
            et.setMutable(false);
        }
        return et;
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        SymbolTable::setCurrentFunction(funcname_);
        visitor.visit(*this);
        ftype_->accept(visitor);
        visitor.postVisit(*this);
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    }
private:
    std::string funcname_;
    FuncTypeNode* ftype_;
};





class FuncAssignNode : public Node
{
public:
    FuncAssignNode(Node* funcstart, Node* funcbody)
        : funcstart_(funcstart), funcbody_(funcbody)
    {
    }
    virtual ~FuncAssignNode()
    {
        delete funcstart_;
        delete funcbody_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        funcstart_->accept(visitor);
        funcbody_->accept(visitor);
        visitor.postVisit(*this);
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
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
    virtual ~FuncArgsNode()
    {
        for (auto arg : args_) {
            delete arg;
        }
    }
    void addArg(Node* expr)
    {
        args_.push_back(expr);
    }
    const std::vector<Node*>& arguments() const
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
    std::vector<Node*> args_;
};




class ReturnStatementNode : public Node
{
public:
    ReturnStatementNode(Node* expr)
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
    Node* expr_;
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
class FuncCallLikeNode : public Node
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        SymbolTable::setCurrentFunction(funcname_);
        visitor.visit(*this);
        funcargs_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    std::string funcname_;
    FuncArgsNode* funcargs_;
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

    const std::string& name() const
    {
    	return funcname_;
    }

    virtual const FuncArgsNode* args() const
    {
    	return funcargs_;
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
        SymbolTable::setCurrentFunction(loop_name_);
        visitor.visit(*this);
        loop_block_->accept(visitor);
        visitor.postVisit(*this);
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    }
private:
    std::string loop_variable_;
    std::string loop_set_;
    std::string loop_name_;
    SequenceNode* loop_block_;
};



class ArrayNode : public Node
{
public:
    ArrayNode(FuncArgsNode* expr_list)
        : expr_list_(expr_list)
    {
        type_ = expr_list->arguments().front()->type();
        type_.setArraySize(expr_list->arguments().size());
    }
    virtual ~ArrayNode()
    {
        delete expr_list_;
    }
    EquelleType type() const
    {
        return type_;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_list_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    FuncArgsNode* expr_list_;
    EquelleType type_;
};



class RandomAccessNode : public Node
{
public:
    RandomAccessNode(Node* expr, const int index)
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
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
private:
    Node* expr_;
    int index_;
};

class StencilAssignmentNode : public Node
{
public:
	StencilAssignmentNode(StencilNode* lhs, Node* rhs)
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
private:
    StencilNode* lhs_;
	Node* rhs_;
};



#endif // ASTNODES_HEADER_INCLUDED
