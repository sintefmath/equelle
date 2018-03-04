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
#include <cmath>



static std::logic_error compilerError(std::string place, std::string msg)
{
    return std::logic_error("Internal compiler error in " + place + ": " + msg);
}

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
    // Only some nodes (ArrayNode, FuncCallNode, VarNode) may legally call this.
    // Therefore it is somewhat of a hack to put it in this base class.
    // Not returning a reference since result may need to be created on the fly.
    virtual std::vector<Dimension> arrayDimension() const
    {
        throw std::logic_error("cannot call arrayDimension() on any ExpressionNode type.");
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
    virtual int numChildren()
    {
        return nodes_.size();
    }
    virtual Node* getChild(const int index)
    {
        return nodes_[index];
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index < nodes_.size() && index >= 0) {
            nodes_[index] = child;
        } else {
            throw compilerError("SequenceNode::setChild()", "Index out of range.");
        }
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
    virtual int numChildren() { return 0; }
    virtual Node* getChild(const int index)
    {
        throw compilerError("NumberNode::getChild()", "NumberNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("NumberNode::setChild()", "NumberNode has no children.");
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
    virtual int numChildren(){ return 0; }
    virtual Node* getChild(const int index)
    {
        throw compilerError("StringNode::getChild()", "StringNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("StringNode::setChild()", "StringNode has no children.");
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
    virtual int numChildren()
    {
        return 0;
    }
    virtual Node* getChild(const int index)
    {
        throw compilerError("TypeNode::getChild()", "TypeNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("TypeNode::setChild()", "TypeNode has no children.");
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
    virtual int numChildren()
    {
        return 3;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return btype_;
            case 1 : return gridmapping_;
            case 2 : return subsetof_;
            default: throw compilerError("CollectionTypeNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : btype_ = dynamic_cast<TypeNode*>(child); break;
            case 1 : gridmapping_ = dynamic_cast<ExpressionNode*>(child); break;
            case 2 : subsetof_ = dynamic_cast<ExpressionNode*>(child); break;
            default: throw compilerError("CollectionTypeNode::setChild()", "Index is out of range.");
        }
    }

private:
    TypeNode* btype_;
    ExpressionNode* gridmapping_;
    ExpressionNode* subsetof_;
};



class ArrayTypeNode : public TypeNode
{
public:
    ArrayTypeNode(TypeNode* btype, const int array_size)
        : TypeNode(EquelleType()),
          btype_(btype),
          array_size_(array_size)
    {
    }
    ~ArrayTypeNode()
    {
        delete btype_;
    }
    const TypeNode* baseType() const
    {
        return btype_;
    }
    EquelleType type() const
    {
        EquelleType bt = btype_->type();
        bt.setArraySize(array_size_);
        return bt;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return btype_;
        } else {
            throw compilerError("ArrayTypeNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0) {
            btype_ = dynamic_cast<TypeNode*>(child);
        } else {
            throw compilerError("ArrayTypeNode::setChild()", "Index is out of range.");
        }
        
    }
private:
    TypeNode* btype_;
    int array_size_;
};



class SequenceTypeNode : public TypeNode
{
public:
    explicit SequenceTypeNode(TypeNode* btype)
        : TypeNode(EquelleType()),
          btype_(btype)
    {
    }
    ~SequenceTypeNode()
    {
        delete btype_;
    }
    const TypeNode* baseType() const
    {
        return btype_;
    }
    EquelleType type() const
    {
        EquelleType bt = btype_->type();
        bt.setCompositeType(Sequence);
        return bt;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return btype_;
        } else {
            throw compilerError("SequenceTypeNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0) {
            btype_ = dynamic_cast<TypeNode*>(child);
        } else {
            throw compilerError("SequenceTypeNode::setChild()", "Index is out of range.");
        }
        
    }
private:
    TypeNode* btype_;
};



class MutableTypeNode : public TypeNode
{
public:
    explicit MutableTypeNode(TypeNode* btype)
        : TypeNode(EquelleType()),
          btype_(btype)
    {
    }
    ~MutableTypeNode()
    {
        delete btype_;
    }
    const TypeNode* baseType() const
    {
        return btype_;
    }
    EquelleType type() const
    {
        EquelleType bt = btype_->type();
        bt.setMutable(true);
        return bt;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return btype_;
        } else {
            throw compilerError("MutableTypeNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0) {
            btype_ = dynamic_cast<TypeNode*>(child);
        } else {
            throw compilerError("MutableTypeNode::setChild()", "Index is out of range.");
        }
        
    }
private:
    TypeNode* btype_;
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
        switch (op_) {
        case Add:
            return lt; // should be identical to rt.
        case Subtract:
            return lt; // should be identical to rt.
        case Multiply: {
            const bool isvec = lt.basicType() == Vector || rt.basicType() == Vector;
            const BasicType bt = isvec ? Vector : Scalar;
            const bool coll = lt.isCollection() || rt.isCollection();
            const bool sequence = lt.isSequence() || rt.isSequence();
            const CompositeType ct = coll ? Collection : (sequence ? Sequence : None);
            const int gm = lt.isCollection() ? lt.gridMapping() : rt.gridMapping();
            return EquelleType(bt, ct, gm);
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
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return left_;
            case 1 : return right_;
            default: throw compilerError("BinaryOpNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : left_ = dynamic_cast<ExpressionNode*>(child); break;
            case 1 : right_ = dynamic_cast<ExpressionNode*>(child); break;
            default: throw compilerError("BinaryOpNode::setChild()", "Index is out of range.");
        }
    }
private:
    BinaryOp op_;
    ExpressionNode* left_;
    ExpressionNode* right_;
};

// Class for the operation a + b * c
class MultiplyAddNode : public ExpressionNode
{
public:
    MultiplyAddNode(ExpressionNode* a, ExpressionNode* b, ExpressionNode* c)
        : a_(a), b_(b), c_(c)
    {
    }
    virtual ~MultiplyAddNode()
    {
        delete a_;
        delete b_;
        delete c_;
    }
    virtual EquelleType type() const
    {

        // Type of left and right side of multiplication
        EquelleType lt = a_->type();
        EquelleType rt = b_->type();

        // This test is taken from BinaryOpNode::type()
        const bool isvec = lt.basicType() == Vector || rt.basicType() == Vector;
        const BasicType bt = isvec ? Vector : Scalar;
        const bool coll = lt.isCollection() || rt.isCollection();
        const bool sequence = lt.isSequence() || rt.isSequence();
        const CompositeType ct = coll ? Collection : (sequence ? Sequence : None);
        const int gm = lt.isCollection() ? lt.gridMapping() : rt.gridMapping();

        auto multiplicationType = EquelleType(bt, ct, gm);

        // We return the type of a * b since c should have the same type
        return multiplicationType;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        a_->accept(visitor);
        visitor.midVisit(*this);
        b_->accept(visitor);
        visitor.midVisit(*this);
        c_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 3;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return a_;
            case 1 : return b_;
            case 2 : return c_;
            default: throw compilerError("MultiplyAddNode::getChild()", "Index out of range.");;
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        ExpressionNode* exprchild = dynamic_cast<ExpressionNode*>(child);
        switch (index) {
            case 0 : a_ = exprchild; break;
            case 1 : b_ = exprchild; break;
            case 2 : c_ = exprchild; break;
            default: throw compilerError("MultiplyAddNode::setChild()", "Index is out of range.");
        }
    }
private:
    ExpressionNode* a_;
    ExpressionNode* b_;
    ExpressionNode* c_;
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
    const ExpressionNode* left() const
    {
        return left_;
    }
    const ExpressionNode* right() const
    {
        return right_;
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
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return left_;
            case 1 : return right_;
            default: throw compilerError("ComparisonOpNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        ExpressionNode* exprchild = dynamic_cast<ExpressionNode*>(child);
        switch (index) {
            case 0 : left_ = exprchild; break;
            case 1 : right_ = exprchild; break;
            default: throw compilerError("ComparisonOpNode::setChild()", "Index is out of range.");
        }
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
    const ExpressionNode* normedExpression() const
    {
        return expr_to_norm_;
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
                throw std::logic_error("internal compiler error in NormNode::dimension().");
            }
            return d;
        }
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_to_norm_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return expr_to_norm_;
        }
        throw compilerError("NormNode::getChild()", "Index is out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        ExpressionNode* exprchild = dynamic_cast<ExpressionNode*>(child);
        if (index == 0) {
            expr_to_norm_ = exprchild;
        }
        throw compilerError("NormNode::setChild()", "Index is out of range.");
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
    const ExpressionNode* negatedExpression() const
    {
        return expr_to_negate_;
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return expr_to_negate_;
        }
        throw compilerError("UnaryNegationNode::getChild()", "Index is out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        
        if (index == 0){
            expr_to_negate_ = dynamic_cast<ExpressionNode*>(child);
        }        
        throw compilerError("UnaryNegationNode::setChild()", "Index is out of range.");
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
    EquelleType leftType() const
    {
        return left_->type();
    }
    EquelleType rightType() const
    {
        return right_->type();
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
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return left_;
            case 1 : return right_;
            default: throw compilerError("OnNode::getChild()", "Index out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        ExpressionNode* exprchild = dynamic_cast<ExpressionNode*>(child);
        switch (index) {
            case 0 : left_ = exprchild; break;
            case 1 : right_ = exprchild; break;
            default: throw compilerError("OnNode::setChild()", "Index out of range.");
        }
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
    const ExpressionNode* predicate() const
    {
        return predicate_;
    }
    const ExpressionNode* ifTrue() const
    {
        return iftrue_;
    }
    const ExpressionNode* ifFalse() const
    {
        return iffalse_;
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
    virtual int numChildren()
    {
        return 3;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return predicate_;
            case 1 : return iftrue_;
            case 2 : return iffalse_;
            default: throw compilerError("TrinaryIfNode::getChild()", "Index is out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : predicate_ = dynamic_cast<ExpressionNode*>(child); break;
            case 1 : iftrue_ = dynamic_cast<ExpressionNode*>(child); break;
            case 2 : iffalse_ = dynamic_cast<ExpressionNode*>(child); break;
            default: throw compilerError("TrinaryIfNode::setChild()", "Index out of range.");
        }
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if(index == 0){
            return type_;
        }
        throw compilerError("VarDeclNode::getChild()", "Index is out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : type_ = dynamic_cast<TypeNode*>(child); break;
            default: throw compilerError("VarDeclNode::setChild()", "Index is out of range.");
        }
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return expr_;
        }
        throw compilerError("VarAssignNode::getChild()", "Index is out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0){
            expr_ = dynamic_cast<ExpressionNode*>(child);
        }else
        {
            throw compilerError("VarAssignNode::setChild()", "Index is out of range.");
        }
    }
private:
    std::string varname_;
    ExpressionNode* expr_;
};




class VarNode : public ExpressionNode
{
public:
    VarNode(const std::string& varname)
        : varname_(varname),
          instantiation_index_(-1)
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
        if (SymbolTable::isVariableDeclared(varname_)) {
            return SymbolTable::variableDimension(varname_);
        } else if (SymbolTable::isFunctionDeclared(varname_)) {
            // Function reference.
            return Dimension();
        } else {
            throw std::logic_error("Internal compiler error in VarNode::dimension().");
        }
    }
    std::vector<Dimension> arrayDimension() const
    {
        if (SymbolTable::isVariableDeclared(varname_)) {
            return SymbolTable::variableArrayDimension(varname_);
        } else {
            throw std::logic_error("Internal compiler error in VarNode::arrayDimension().");
        }
    }
    const std::string& name() const
    {
        return varname_;
    }
    int instantiationIndex() const
    {
        return instantiation_index_;
    }
    void setInstantiationIndex(const int index)
    {
        instantiation_index_ = index;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
    virtual int numChildren()
    {
        return 0;
    }
    virtual Node* getChild(const int index)
    {
        throw compilerError("VarNode::getChild()", "VarNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("VarNode::setChild()", "VarNode has no children.");
    }
private:
    std::string varname_;
    int instantiation_index_;
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
    virtual int numChildren()
    {
        return decls_.size();
    }
    virtual Node* getChild(const int index)
    {
        return decls_[index];
    }
    virtual void setChild(const int index, Node* child)
    {
        decls_[index] = dynamic_cast<VarDeclNode*>(child);
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
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        if ( index == 0 ) {
            return argtypes_;
        } else 
        if ( index == 1 ) {
            return rtype_;
        } 
        throw compilerError("FuncTypeNode::getChild()", "Index is out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if ( index == 0 ) {
            argtypes_ = dynamic_cast<FuncArgsDeclNode*>(child);
        }else
        if ( index == 1 ) {
            rtype_ = dynamic_cast<TypeNode*>(child);
        }
        throw compilerError("FuncTypeNode::setChild()", "Index is out of range.");
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
    virtual int numChildren()
    {
        return 0;
    }
    virtual Node* getChild(const int index)
    {
        throw compilerError("FuncRefNode::getChild()", "FuncRefNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("FuncRefNode::setChild()", "FuncRefNode has no children.");
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
    virtual int numChildren()
    {
        return 0;
    }
    virtual Node* getChild(const int index)
    {
        throw compilerError("JustAnIdentifierNode::getChild()", "JustAnIdentifierNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("JustAnIdentifierNode::setChild()", "JustAnIdentifierNode has no children.");
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if ( index == 0 ){
            return ftype_;
        }
        throw compilerError("FuncDeclNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if ( index == 0 ){
            ftype_ = dynamic_cast<FuncTypeNode*>(child);
        } else {
            throw compilerError("FuncDeclNode::setChild()", "Index out of range.");
        }
        
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
    virtual int numChildren()
    {
        return args_.size();
    }
    virtual Node* getChild(const int index)
    {
        return args_[index];
    }
    virtual void setChild(const int index, Node* child)
    {
        args_[index] = dynamic_cast<ExpressionNode*>(child);
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
    Dimension dimension() const
    {
        return expr_->dimension();
    }
    std::vector<Dimension> arrayDimension() const
    {
        return expr_->arrayDimension();
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return expr_;
        }
        throw compilerError("ReturnStatementNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0){
            expr_ = dynamic_cast<ExpressionNode*>(child);
        }
        throw compilerError("ReturnStatementNode::setChild()", "Index out of range.");
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return funcargs_;
        }
        throw compilerError("FuncStartNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0){
            funcargs_ = dynamic_cast<FuncArgsNode*>(child);
        }
        throw compilerError("FuncStartNode::setChild()", "Index out of range.");
        
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
    const FuncArgsNode* args() const
    {
        return funcstart_->args();
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
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return funcstart_;
        }
        throw compilerError("FuncAssignNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : funcbody_ = dynamic_cast<FuncStartNode*>(child); break;
            case 1 : funcbody_ = child; break;
            default: throw compilerError("FuncAssignNode::setChild()", "Index out of range.");
        }       
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return args_;
        }
        throw compilerError("StencilNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if( index == 0 ) {
            args_ = dynamic_cast<FuncArgsNode*>(child);
        }
        throw compilerError("StencilNode::setChild()", "Index out of range.");
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
        : funcname_(funcname),
          funcargs_(funcargs),
          dsr_(dynamic_subset_return),
          instantiation_index_(-999)
    {}

    virtual ~FuncCallNode()
    {
        delete funcargs_;
    }

    void setDynamicSubsetReturn(const int dynamic_subset_return)
    {
        dsr_ = dynamic_subset_return;
    }

    void setReturnType(const EquelleType& return_type)
    {
        return_type_ = return_type;
    }

    void setInstantiationIndex(const int instantiation_index)
    {
        instantiation_index_ = instantiation_index;
    }

    int instantiationIndex() const
    {
        return instantiation_index_;
    }

    EquelleType type() const
    {
        if (!return_type_.isInvalid()) {
            return return_type_;
        }
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
        if (type().isArray() || dimension_.size() != 1) {
            throw std::logic_error("Internal compiler error in FuncCallNode::dimension().");
        }
        return dimension_[0];
    }

    std::vector<Dimension> arrayDimension() const
    {
        if (!type().isArray()) {
            throw std::logic_error("Internal compiler error in FuncCallNode::dimension().");
        }
        return dimension_;
    }

    void setDimension(const std::vector<Dimension>& dims)
    {
        dimension_ = dims;
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return funcargs_;
            default: throw compilerError("FuncCallNode::getChild()", "Index out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : funcargs_ = dynamic_cast<FuncArgsNode*>(child); break;
            default: throw compilerError("FuncCallNode::setChild()", "Index out of range.");
        }
    }
private:
    std::string funcname_;
    FuncArgsNode* funcargs_;
    int dsr_;
    // return_type_ is only set for template instantiation type calls.
    EquelleType return_type_;
    // dimension_ should always be set by the checking visitor.
    std::vector<Dimension> dimension_;
    int instantiation_index_;
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return func_call_;
        }
        throw compilerError("FuncCallStatementNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if( index == 0 ) {
            func_call_ = dynamic_cast<FuncCallNode*>(child);
        }
        throw compilerError("FuncCallStatementNode::setChild()", "Index out of range.");
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
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return loop_block_;
        }
        throw compilerError("LoopNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if( index == 0 ) {
            loop_block_ = dynamic_cast<SequenceNode*>(child);
        }
        throw compilerError("LoopNode::setChild()", "Index out of range.");
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
    const FuncArgsNode* expressionList()
    {
        return expr_list_;
    }
    EquelleType type() const
    {
        EquelleType t = expr_list_->arguments().front()->type();
        t.setArraySize(expr_list_->arguments().size());
        return t;
    }
    Dimension dimension() const
    {
        throw compilerError("ArrayNode::dimension()", "Meaningless to ask for array dimension since array elements may have different dimension.");
        return Dimension();
    }
    std::vector<Dimension> arrayDimension() const
    {
        const int size = expr_list_->arguments().size();
        std::vector<Dimension> dims(size);
        for (int elem = 0; elem < size; ++elem) {
            dims[elem] = expr_list_->arguments()[elem]->dimension();
        }
        return dims;
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_list_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0){
            return expr_list_;
        }
        throw compilerError("ArrayNode::getChild()", "Index out of range.");

    }
    virtual void setChild(const int index, Node* child)
    {
        if( index == 0 ) {
            expr_list_ = dynamic_cast<FuncArgsNode*>(child);
        }
        throw compilerError("ArrayNode::setChild()", "Index out of range.");
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
    const ExpressionNode* expressionToAccess() const
    {
        return expr_;
    }
    EquelleType type() const
    {
        // Either expr_ must be an Array, or, if not,
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
        if (expr_->type().isArray()) {
            return expr_->arrayDimension()[index_];
        } else {
            return expr_->dimension();
        }
    }
    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        expr_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if (index == 0) {
            return expr_;
        }
        throw compilerError("RandomAccessNode::getChild()", "Index out of range.");
    }
    virtual void setChild(const int index, Node* child)
    {
        if (index == 0) {
            expr_ = dynamic_cast<ExpressionNode*>(child);
        }
        throw compilerError("RandomAccessNode::setChild()", "Index out of range.");
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

    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return lhs_;
            case 1 : return rhs_;
            default: throw compilerError("StencilAssignmentNode::getChild()", "Index out of range.");
        }
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : lhs_ = dynamic_cast<StencilNode*>(child); break;
            case 1 : rhs_ = dynamic_cast<ExpressionNode*>(child); break;
            default: throw compilerError("StencilAssignmentNode::setChild()", "Index out of range.");
        }
    }
private:
    StencilNode* lhs_;
    ExpressionNode* rhs_;
};



class UnitNode : public Node
{
public:
    // Dimension
    virtual Dimension dimension() const = 0;

    // The number you must multiply a quantity given in the
    // current unit with to obtain an SI quantity.
    // For example for Inch, the factor ie 0.0254.
    virtual double conversionFactorSI() const = 0;
    
    virtual int numChildren()
    {
        return 0;
    }
    virtual Node* getChild(const int index)
    {
        throw compilerError("UnitNode::getChild()", "UnitNode has no children.");
    }
    virtual void setChild(const int index, Node* child)
    {
        throw compilerError("UnitNode::setChild()", "UnitNode has no children.");
    }

};


class BasicUnitNode : public UnitNode
{
public:
    BasicUnitNode(const std::string& name)
        : conv_factor_(-1e100)
    {
        UnitData ud = unitFromString(name);
        if (ud.valid) {
            dimension_ = ud.dimension;
            conv_factor_ = ud.conv_factor;
        } else {
            std::string err = "Unit name not recognised: ";
            err += name;
            throw std::runtime_error(err.c_str());
        }
    }

    BasicUnitNode(const Dimension dimension_arg,
                  const double conversion_factor_SI)
        : dimension_(dimension_arg),
          conv_factor_(conversion_factor_SI)
    {
    }

    Dimension dimension() const
    {
        return dimension_;
    }

    double conversionFactorSI() const
    {
        return conv_factor_;
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
    }
private:
    Dimension dimension_;
    double conv_factor_;
};



class BinaryOpUnitNode : public UnitNode
{
public:
    BinaryOpUnitNode(BinaryOp op, UnitNode* left, UnitNode* right)
        : op_(op),
          left_(left),
          right_(right)
    {
    }

    ~BinaryOpUnitNode()
    {
        delete left_;
        delete right_;
    }

    BinaryOp op() const
    {
        return op_;
    }

    Dimension dimension() const
    {
        switch (op_) {
        case Multiply:
            return left_->dimension() + right_->dimension();
        case Divide:
            return left_->dimension() - right_->dimension();
        default:
            throw std::logic_error("Units can only be manipulated with '*', '/' or '^'.");
        }
    }

    double conversionFactorSI() const
    {
        switch (op_) {
        case Multiply:
            return left_->conversionFactorSI() * right_->conversionFactorSI();
        case Divide:
            return left_->conversionFactorSI() / right_->conversionFactorSI();
        default:
            throw std::logic_error("Units can only be manipulated with '*', '/' or '^'.");
        }
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        left_->accept(visitor);
        visitor.midVisit(*this);
        right_->accept(visitor);
        visitor.postVisit(*this);
    }
    
    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return left_;
            case 1 : return right_;
            default: throw compilerError("BinaryOpUnitNode::getChild()","Index out of range.");
        }
        return left_;
        return right_;
    }
    virtual void setChild(const int index, Node* child)
    {
        switch (index) {
            case 0 : left_ = dynamic_cast<UnitNode*>(child); break;
            case 1 : right_ = dynamic_cast<UnitNode*>(child); break;
            default: throw compilerError("BinaryOpUnitNode::setChild()","Index out of range.");
        }
    }
private:
    BinaryOp op_;
    UnitNode* left_;
    UnitNode* right_;
};




class PowerUnitNode : public UnitNode
{
public:
    PowerUnitNode(UnitNode* unit, int power)
        : unit_(unit),
          power_(power)
    {
    }

    ~PowerUnitNode()
    {
        delete unit_;
    }

    int power() const
    {
        return power_;
    }

    Dimension dimension() const
    {
        return unit_->dimension() * power_;
    }

    double conversionFactorSI() const
    {
        return std::pow(unit_->conversionFactorSI(), power_);
    }

    virtual void accept(ASTVisitorInterface& visitor)
    {
        visitor.visit(*this);
        unit_->accept(visitor);
        visitor.postVisit(*this);
    }
    virtual int numChildren()
    {
        return 1;
    }
    virtual Node* getChild(const int index)
    {
        if ( index == 0 ) {
            return unit_;
        }
        throw compilerError("PowerUnitNode::getChild()","Index out of range.");
        
    }
    virtual void setChild(const int index, Node* child)
    {
        if ( index == 0 ) {
            unit_ = dynamic_cast<UnitNode*>(child);
            return;
        }
        throw compilerError("PowerUnitNode::getChild()","Index out of range.");
    }

private:
    UnitNode* unit_;
    int power_;
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

    virtual int numChildren()
    {
        return 2;
    }
    virtual Node* getChild(const int index)
    {
        switch (index) {
            case 0 : return number_;
            case 1 : return unit_;
            default: throw compilerError("QuantityUnitNode::getChild()","Index out of range.");
        }
        
    }
    virtual void setChild(const int index, Node* child)
    {
        switch(index) {
            case 0: number_ = dynamic_cast<NumberNode*>(child); break;
            case 1: unit_ = dynamic_cast<UnitNode*>(child); break;
            default: throw compilerError("QuantityUnitNode::setChild()","Index out of range.");
        }
    }

private:
    NumberNode* number_;
    UnitNode* unit_;
};


#endif // ASTNODES_HEADER_INCLUDED
