/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintEquelleASTVisitor.hpp"
#include "ASTNodes.hpp"
#include <iostream>




PrintEquelleASTVisitor::PrintEquelleASTVisitor()
    : indent_(0)
{
}

PrintEquelleASTVisitor::~PrintEquelleASTVisitor()
{
}





void PrintEquelleASTVisitor::visit(SequenceNode& node)
{
}

void PrintEquelleASTVisitor::postVisit(SequenceNode& node)
{
}

void PrintEquelleASTVisitor::visit(NumberNode& node)
{
    std::cout.precision(16);
    std::cout << node.number();
}

void PrintEquelleASTVisitor::visit(TypeNode& node)
{
    std::cout << node.type().equelleString();
}

void PrintEquelleASTVisitor::visit(FuncTypeNode& node)
{
    std::cout << node.funcType().equelleString();
}

void PrintEquelleASTVisitor::visit(BinaryOpNode& node)
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
    std::cout << "[[ " << op << " ";
}

void PrintEquelleASTVisitor::postVisit(BinaryOpNode& node)
{
    std::cout << " ]]";
}

void PrintEquelleASTVisitor::visit(NormNode& node)
{
    std::cout << '|';
}

void PrintEquelleASTVisitor::postVisit(NormNode& node)
{
    std::cout << '|';
}

void PrintEquelleASTVisitor::visit(UnaryNegationNode& node)
{
    std::cout << '-';
}

void PrintEquelleASTVisitor::postVisit(UnaryNegationNode& node)
{
}

void PrintEquelleASTVisitor::visit(OnNode& node)
{
    std::cout << "[[ On ";
}

void PrintEquelleASTVisitor::postVisit(OnNode& node)
{
    std::cout << " ]]";
}

void PrintEquelleASTVisitor::visit(TrinaryIfNode& node)
{
    std::cout << "[[ ?: ";
}

void PrintEquelleASTVisitor::postVisit(TrinaryIfNode& node)
{
    std::cout << " ]]";
}

void PrintEquelleASTVisitor::visit(VarDeclNode& node)
{
    std::cout << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(VarDeclNode& node)
{
    endl();
}

void PrintEquelleASTVisitor::visit(VarAssignNode& node)
{
    std::cout << node.name() << " = ";
}

void PrintEquelleASTVisitor::postVisit(VarAssignNode& node)
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

void PrintEquelleASTVisitor::visit(FuncArgsDeclNode& node)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintEquelleASTVisitor::postVisit(FuncArgsDeclNode& node)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintEquelleASTVisitor::visit(FuncDeclNode& node)
{
    std::cout << node.name() << " : ";
}

void PrintEquelleASTVisitor::postVisit(FuncDeclNode& node)
{
    endl();
}

void PrintEquelleASTVisitor::visit(FuncStartNode& node)
{
    std::cout << node.name() << '(';
}

void PrintEquelleASTVisitor::postVisit(FuncStartNode& node)
{
    std::cout << ") = {";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncAssignNode& node)
{
    ++indent_;
}

void PrintEquelleASTVisitor::postVisit(FuncAssignNode& node)
{
    std::cout << "}";
    endl();
}

void PrintEquelleASTVisitor::visit(FuncArgsNode& node)
{
    // std::cout << "{FuncArgsNode::visit()}";
}

void PrintEquelleASTVisitor::postVisit(FuncArgsNode& node)
{
    // std::cout << "{FuncArgsNode::postVisit()}";
}

void PrintEquelleASTVisitor::visit(ReturnStatementNode& node)
{
    std::cout << "-> ";
}

void PrintEquelleASTVisitor::postVisit(ReturnStatementNode& node)
{
    --indent_;
    endl();
}

void PrintEquelleASTVisitor::visit(FuncCallNode& node)
{
    std::cout << node.name() << '(';
}

void PrintEquelleASTVisitor::postVisit(FuncCallNode& node)
{
    std::cout << ')';
}



void PrintEquelleASTVisitor::endl() const
{
    std::cout << '\n' << indent();
}

std::string PrintEquelleASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}
