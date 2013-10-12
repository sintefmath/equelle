/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintEquelleASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>




PrintEquelleASTVisitor::PrintEquelleASTVisitor()
    : indent_(0)
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

void PrintEquelleASTVisitor::visit(TypeNode& node)
{
    std::cout << SymbolTable::equelleString(node.type());
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

void PrintEquelleASTVisitor::midVisit(OnNode&)
{
    std::cout << " On ";
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
    std::cout << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(VarDeclNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(VarAssignNode& node)
{
    std::cout << node.name() << " = ";
}

void PrintEquelleASTVisitor::postVisit(VarAssignNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(VarNode& node)
{
    std::cout << node.name();
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
    std::cout << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(FuncDeclNode&)
{
    endl();
}

void PrintEquelleASTVisitor::visit(FuncStartNode& node)
{
    std::cout << node.name() << '(';
}

void PrintEquelleASTVisitor::postVisit(FuncStartNode&)
{
    std::cout << ") = {";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncAssignNode&)
{
    ++indent_;
}

void PrintEquelleASTVisitor::postVisit(FuncAssignNode&)
{
    std::cout << "}";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncArgsNode&)
{
    // std::cout << "{FuncArgsNode::visit()}";
}

void PrintEquelleASTVisitor::midVisit(FuncArgsNode&)
{
    // std::cout << "{FuncArgsNode::midVisit()}";
}

void PrintEquelleASTVisitor::postVisit(FuncArgsNode&)
{
    // std::cout << "{FuncArgsNode::postVisit()}";
}

void PrintEquelleASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << "-> ";
}

void PrintEquelleASTVisitor::postVisit(ReturnStatementNode&)
{
    --indent_;
    endl();
}

void PrintEquelleASTVisitor::visit(FuncCallNode& node)
{
    std::cout << node.name() << '(';
}

void PrintEquelleASTVisitor::postVisit(FuncCallNode&)
{
    std::cout << ')';
}

void PrintEquelleASTVisitor::visit(FuncCallStatementNode&)
{
}

void PrintEquelleASTVisitor::postVisit(FuncCallStatementNode&)
{
    endl();
}



void PrintEquelleASTVisitor::endl() const
{
    std::cout << '\n' << indent();
}

std::string PrintEquelleASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}
