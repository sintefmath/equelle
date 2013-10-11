/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintASTVisitor.hpp"
#include "ASTNodes.hpp"
#include <iostream>




PrintASTVisitor::PrintASTVisitor()
    : indent_(0)
{
}

PrintASTVisitor::~PrintASTVisitor()
{
}





void PrintASTVisitor::visit(SequenceNode& node)
{
    std::cout << indent() << "SequenceNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(NumberNode& node)
{
    std::cout << indent() << "NumberNode: " << node.number() << '\n';
}

void PrintASTVisitor::visit(TypeNode& node)
{
    std::cout << indent() << "TypeNode: " << node.type().equelleString() << '\n';
}

void PrintASTVisitor::visit(FuncTypeNode& node)
{
    std::cout << indent() << "FuncTypeNode: " << node.funcType().equelleString() << '\n';
}

void PrintASTVisitor::visit(BinaryOpNode& node)
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
    std::cout << indent() << "BinaryOpNode: " << op << '\n';

    ++indent_;
}

void PrintASTVisitor::visit(NormNode& node)
{
    std::cout << indent() << "NormNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(UnaryNegationNode& node)
{
    std::cout << indent() << "UnaryNegationNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(OnNode& node)
{
    std::cout << indent() << "OnNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(TrinaryIfNode& node)
{
    std::cout << indent() << "TrinaryIfNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(VarDeclNode& node)
{
    std::cout << indent() << "VarDeclNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(VarAssignNode& node)
{
    std::cout << indent() << "VarAssignNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(VarNode& node)
{
    std::cout << indent() << "VarNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(FuncRefNode& node)
{
    std::cout << indent() << "FuncRefNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(JustAnIdentifierNode& node)
{
    std::cout << indent() << "JustAnIdentifierNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(FuncArgsDeclNode& node)
{
    std::cout << indent() << "FuncArgsDeclNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncDeclNode& node)
{
    std::cout << indent() << "FuncDeclNod: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(FuncStartNode& node)
{
    std::cout << indent() << "FuncStartNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(FuncAssignNode& node)
{
    std::cout << indent() << "FuncAssignNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncArgsNode& node)
{
    std::cout << indent() << "FuncArgsNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(ReturnStatementNode& node)
{
    std::cout << indent() << "ReturnStatementNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncCallNode& node)
{
    std::cout << indent() << "FuncCallNode: " << node.name() << '\n';
    ++indent_;
}




void PrintASTVisitor::postVisit(SequenceNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(BinaryOpNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(NormNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(UnaryNegationNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(OnNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(TrinaryIfNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(VarDeclNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(VarAssignNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncArgsDeclNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncDeclNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncStartNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncAssignNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncArgsNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(ReturnStatementNode& node)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncCallNode& node)
{
    --indent_;
}




std::string PrintASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}
