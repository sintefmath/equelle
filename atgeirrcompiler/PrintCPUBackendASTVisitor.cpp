/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintCPUBackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>




PrintCPUBackendASTVisitor::PrintCPUBackendASTVisitor()
    : indent_(0)
{
}

PrintCPUBackendASTVisitor::~PrintCPUBackendASTVisitor()
{
}





void PrintCPUBackendASTVisitor::visit(SequenceNode&)
{
}

void PrintCPUBackendASTVisitor::postVisit(SequenceNode&)
{
}

void PrintCPUBackendASTVisitor::visit(NumberNode& node)
{
    std::cout.precision(16);
    std::cout << node.number();
}

void PrintCPUBackendASTVisitor::visit(TypeNode& node)
{
    std::cout << SymbolTable::equelleString(node.type());
}

void PrintCPUBackendASTVisitor::visit(FuncTypeNode& node)
{
    std::cout << node.funcType().equelleString();
}

void PrintCPUBackendASTVisitor::visit(BinaryOpNode&)
{
    std::cout << '(';
}

void PrintCPUBackendASTVisitor::midVisit(BinaryOpNode& node)
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

void PrintCPUBackendASTVisitor::postVisit(BinaryOpNode&)
{
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(NormNode&)
{
    std::cout << '|';
}

void PrintCPUBackendASTVisitor::postVisit(NormNode&)
{
    std::cout << '|';
}

void PrintCPUBackendASTVisitor::visit(UnaryNegationNode&)
{
    std::cout << '-';
}

void PrintCPUBackendASTVisitor::postVisit(UnaryNegationNode&)
{
}

void PrintCPUBackendASTVisitor::visit(OnNode&)
{
    std::cout << '(';
}

void PrintCPUBackendASTVisitor::midVisit(OnNode&)
{
    std::cout << " On ";
}

void PrintCPUBackendASTVisitor::postVisit(OnNode&)
{
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(TrinaryIfNode&)
{
    std::cout << '(';
}

void PrintCPUBackendASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
    std::cout << " ? ";
}

void PrintCPUBackendASTVisitor::colonVisit(TrinaryIfNode&)
{
    std::cout << " : ";
}

void PrintCPUBackendASTVisitor::postVisit(TrinaryIfNode&)
{
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(VarDeclNode& node)
{
    std::cout << node.name() << " : ";
}

void PrintCPUBackendASTVisitor::postVisit(VarDeclNode&)
{
    endl();
}

void PrintCPUBackendASTVisitor::visit(VarAssignNode& node)
{
    std::cout << node.name() << " = ";
}

void PrintCPUBackendASTVisitor::postVisit(VarAssignNode&)
{
    endl();
}

void PrintCPUBackendASTVisitor::visit(VarNode& node)
{
    std::cout << node.name();
}

void PrintCPUBackendASTVisitor::visit(FuncRefNode& node)
{
    std::cout << node.name();
}

void PrintCPUBackendASTVisitor::visit(JustAnIdentifierNode& node)
{
    std::cout << node.name();
}

void PrintCPUBackendASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintCPUBackendASTVisitor::postVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintCPUBackendASTVisitor::visit(FuncDeclNode& node)
{
    std::cout << node.name() << " : ";
}

void PrintCPUBackendASTVisitor::postVisit(FuncDeclNode&)
{
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncStartNode& node)
{
    std::cout << node.name() << '(';
}

void PrintCPUBackendASTVisitor::postVisit(FuncStartNode&)
{
    std::cout << ") = {";
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncAssignNode&)
{
    ++indent_;
}

void PrintCPUBackendASTVisitor::postVisit(FuncAssignNode&)
{
    std::cout << "}";
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncArgsNode&)
{
    // std::cout << "{FuncArgsNode::visit()}";
}

void PrintCPUBackendASTVisitor::postVisit(FuncArgsNode&)
{
    // std::cout << "{FuncArgsNode::postVisit()}";
}

void PrintCPUBackendASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << "-> ";
}

void PrintCPUBackendASTVisitor::postVisit(ReturnStatementNode&)
{
    --indent_;
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncCallNode& node)
{
    std::cout << node.name() << '(';
}

void PrintCPUBackendASTVisitor::postVisit(FuncCallNode&)
{
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(FuncCallStatementNode&)
{
}

void PrintCPUBackendASTVisitor::postVisit(FuncCallStatementNode&)
{
    endl();
}



void PrintCPUBackendASTVisitor::endl() const
{
    std::cout << '\n' << indent();
}

std::string PrintCPUBackendASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}
