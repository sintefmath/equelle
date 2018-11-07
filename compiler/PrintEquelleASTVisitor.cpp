/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintEquelleASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <stdexcept>




PrintEquelleASTVisitor::PrintEquelleASTVisitor()
    : suppressed_(false), indent_(0)
{
}

PrintEquelleASTVisitor::~PrintEquelleASTVisitor()
{
}





void PrintEquelleASTVisitor::visit(SequenceNode&)
{
}

void PrintEquelleASTVisitor::midVisit(SequenceNode&)
{
}

void PrintEquelleASTVisitor::postVisit(SequenceNode&)
{
}

void PrintEquelleASTVisitor::visit(NumberNode& node)
{
    std::cout.precision(16);
    std::cout << node.number();
}

void PrintEquelleASTVisitor::visit(QuantityNode& node)
{
    const double cf = node.conversionFactorSI();
    if (cf != 1.0) {
        std::cout.precision(16);
        std::cout << "(" << cf << " * ";
    }
}

void PrintEquelleASTVisitor::postVisit(QuantityNode& node)
{
    if (node.dimension() != Dimension()) {
        std::cout << ' ' << node.dimension();
    }
    if (node.conversionFactorSI() != 1.0) {
        std::cout << ")";
    }
}

void PrintEquelleASTVisitor::visit(StringNode& node)
{
    std::cout << node.content();
}

void PrintEquelleASTVisitor::visit(TypeNode& node)
{
    std::cout << SymbolTable::equelleString(node.type());
}

void PrintEquelleASTVisitor::visit(CollectionTypeNode& node)
{
    std::cout << SymbolTable::equelleString(node.type());
    suppress();
}

void PrintEquelleASTVisitor::postVisit(CollectionTypeNode& node)
{
    unsuppress();
}

void PrintEquelleASTVisitor::visit(FuncTypeNode& node)
{
    std::cout << node.funcType().equelleString();
}

void PrintEquelleASTVisitor::visit(BinaryOpNode&)
{
    std::cout << '(';
}

void PrintEquelleASTVisitor::midVisit(BinaryOpNode& node)
{
    char op = ' ';
    switch (node.op()) {
    case Add:
        op = '+';
        break;
    case Subtract:
        op = '-';
        break;
    case Multiply:
        op = '*';
        break;
    case Divide:
        op = '/';
        break;
    default:
        break;
    }
    std::cout << ' ' << op << ' ';
}

void PrintEquelleASTVisitor::postVisit(BinaryOpNode&)
{
    std::cout << ')';
}

void PrintEquelleASTVisitor::visit(MultiplyAddNode& node)
{
}

void PrintEquelleASTVisitor::midVisit(MultiplyAddNode& node)
{
}

void PrintEquelleASTVisitor::postVisit(MultiplyAddNode& node)
{
}

void PrintEquelleASTVisitor::visit(MultiplyDivideNode& node)
{
}

void PrintEquelleASTVisitor::midVisit(MultiplyDivideNode& node)
{
}

void PrintEquelleASTVisitor::postVisit(MultiplyDivideNode& node)
{
}

void PrintEquelleASTVisitor::visit(ComparisonOpNode&)
{
    std::cout << '(';
}

void PrintEquelleASTVisitor::midVisit(ComparisonOpNode& node)
{
    std::string op(" ");
    switch (node.op()) {
    case Less:
        op = "<";
        break;
    case Greater:
        op = ">";
        break;
    case LessEqual:
        op = "<=";
        break;
    case GreaterEqual:
        op = ">=";
        break;
    case Equal:
        op = "==";
        break;
    case NotEqual:
        op = "!=";
        break;
    default:
        break;
    }
    std::cout << ' ' << op << ' ';
}

void PrintEquelleASTVisitor::postVisit(ComparisonOpNode&)
{
    std::cout << ')';
}

void PrintEquelleASTVisitor::visit(NormNode&)
{
    std::cout << '|';
}

void PrintEquelleASTVisitor::postVisit(NormNode&)
{
    std::cout << '|';
}

void PrintEquelleASTVisitor::visit(UnaryNegationNode&)
{
    std::cout << '-';
}

void PrintEquelleASTVisitor::postVisit(UnaryNegationNode&)
{
}

void PrintEquelleASTVisitor::visit(OnNode&)
{
    std::cout << '(';
}

void PrintEquelleASTVisitor::midVisit(OnNode& node)
{
    if (node.isExtend()) {
        std::cout << " Extend ";
    } else {
        std::cout << " On ";
    }
}

void PrintEquelleASTVisitor::postVisit(OnNode&)
{
    std::cout << ')';
}

void PrintEquelleASTVisitor::visit(TrinaryIfNode&)
{
    std::cout << '(';
}

void PrintEquelleASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
    std::cout << " ? ";
}

void PrintEquelleASTVisitor::colonVisit(TrinaryIfNode&)
{
    std::cout << " : ";
}

void PrintEquelleASTVisitor::postVisit(TrinaryIfNode&)
{
    std::cout << ')';
}

void PrintEquelleASTVisitor::visit(VarDeclNode& node)
{
    std::cout << indent() << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(VarDeclNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(VarAssignNode& node)
{
    std::cout << indent() << node.name() << " = ";
}

void PrintEquelleASTVisitor::postVisit(VarAssignNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(VarNode& node)
{
    if (!suppressed_) {
        std::cout << node.name();
    }
}

void PrintEquelleASTVisitor::visit(FuncRefNode& node)
{
    std::cout << node.name();
}

void PrintEquelleASTVisitor::visit(JustAnIdentifierNode& node)
{
    std::cout << node.name();
}

void PrintEquelleASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintEquelleASTVisitor::midVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintEquelleASTVisitor::postVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintEquelleASTVisitor::visit(FuncDeclNode& node)
{
    std::cout << indent() << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(FuncDeclNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(FuncStartNode& node)
{
    std::cout << indent() << node.name() << '(';
    ++indent_;
}

void PrintEquelleASTVisitor::postVisit(FuncStartNode&)
{
    std::cout << ") = {";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncAssignNode&)
{
}

void PrintEquelleASTVisitor::postVisit(FuncAssignNode&)
{
    --indent_;
    std::cout << indent() << "}";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncArgsNode&)
{
}

void PrintEquelleASTVisitor::midVisit(FuncArgsNode&)
{
        std::cout << ", ";
}

void PrintEquelleASTVisitor::postVisit(FuncArgsNode&)
{
}

void PrintEquelleASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << indent() << "-> ";
}

void PrintEquelleASTVisitor::postVisit(ReturnStatementNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(FuncCallNode& node)
{
    if (!suppressed_) {
        std::cout << node.name() << '(';
    }
}

void PrintEquelleASTVisitor::postVisit(FuncCallNode&)
{
    if (!suppressed_) {
        std::cout << ')';
    }
}

void PrintEquelleASTVisitor::visit(FuncCallStatementNode&)
{
    std::cout << indent();
}

void PrintEquelleASTVisitor::postVisit(FuncCallStatementNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(LoopNode& node)
{
    SymbolTable::setCurrentFunction(node.loopName());
    std::cout << indent() << "For " << node.loopVariable() << " In " << node.loopSet() << " {";
    ++indent_;
    endl();
}

void PrintEquelleASTVisitor::postVisit(LoopNode&)
{
    --indent_;
    std::cout << indent() << "}";
    endl();
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
}

void PrintEquelleASTVisitor::visit(ArrayNode&)
{
    std::cout << '[';
}

void PrintEquelleASTVisitor::postVisit(ArrayNode&)
{
    std::cout << ']';
}

void PrintEquelleASTVisitor::visit(RandomAccessNode&)
{
}

void PrintEquelleASTVisitor::postVisit(RandomAccessNode& node)
{
    std::cout << "[" << node.index() << "]";
}

void PrintEquelleASTVisitor::visit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintEquelleASTVisitor::midVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintEquelleASTVisitor::postVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintEquelleASTVisitor::visit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintEquelleASTVisitor::postVisit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintEquelleASTVisitor::endl() const
{
    std::cout << '\n';
}

std::string PrintEquelleASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}

void PrintEquelleASTVisitor::suppress()
{
    suppressed_ = true;
}

void PrintEquelleASTVisitor::unsuppress()
{
    suppressed_ = false;
}
