/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
*/

#include "CheckASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <stdexcept>


CheckASTVisitor::CheckASTVisitor()
{
}

CheckASTVisitor::~CheckASTVisitor()
{
}


void CheckASTVisitor::visit(SequenceNode&)
{
}

void CheckASTVisitor::visit(NumberNode& node)
{
}

void CheckASTVisitor::visit(QuantityNode& node)
{
}

void CheckASTVisitor::visit(UnitNode& node)
{
}

void CheckASTVisitor::visit(StringNode& node)
{
}

void CheckASTVisitor::visit(TypeNode& node)
{
}

void CheckASTVisitor::visit(FuncTypeNode& node)
{
}

void CheckASTVisitor::visit(BinaryOpNode& node)
{
}

void CheckASTVisitor::visit(ComparisonOpNode& node)
{
}

void CheckASTVisitor::visit(NormNode&)
{
}

void CheckASTVisitor::visit(UnaryNegationNode&)
{
}

void CheckASTVisitor::visit(OnNode& node)
{
}

void CheckASTVisitor::visit(TrinaryIfNode&)
{
}

void CheckASTVisitor::visit(VarDeclNode& node)
{
}

void CheckASTVisitor::visit(VarAssignNode& node)
{
}

void CheckASTVisitor::visit(VarNode& node)
{
}

void CheckASTVisitor::visit(FuncRefNode& node)
{
}

void CheckASTVisitor::visit(JustAnIdentifierNode& node)
{
}

void CheckASTVisitor::visit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::visit(FuncDeclNode& node)
{
}

void CheckASTVisitor::visit(FuncStartNode& node)
{
}

void CheckASTVisitor::visit(FuncAssignNode&)
{
}

void CheckASTVisitor::visit(FuncArgsNode&)
{
}

void CheckASTVisitor::visit(ReturnStatementNode&)
{
}

void CheckASTVisitor::visit(FuncCallNode& node)
{
}

void CheckASTVisitor::visit(FuncCallStatementNode&)
{
}


void CheckASTVisitor::visit(LoopNode& node)
{
}


void CheckASTVisitor::visit(ArrayNode& node)
{
}


void CheckASTVisitor::visit(RandomAccessNode& node)
{
}




void CheckASTVisitor::midVisit(SequenceNode&)
{
}

void CheckASTVisitor::postVisit(SequenceNode&)
{
}

void CheckASTVisitor::postVisit(QuantityNode&)
{
}

void CheckASTVisitor::midVisit(BinaryOpNode&)
{
}

void CheckASTVisitor::postVisit(BinaryOpNode&)
{
}

void CheckASTVisitor::midVisit(ComparisonOpNode&)
{
}

void CheckASTVisitor::postVisit(ComparisonOpNode&)
{
}

void CheckASTVisitor::postVisit(NormNode&)
{
}

void CheckASTVisitor::postVisit(UnaryNegationNode&)
{
}

void CheckASTVisitor::midVisit(OnNode&)
{
}

void CheckASTVisitor::postVisit(OnNode&)
{
}

void CheckASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
}

void CheckASTVisitor::colonVisit(TrinaryIfNode&)
{
}

void CheckASTVisitor::postVisit(TrinaryIfNode&)
{
}

void CheckASTVisitor::postVisit(VarDeclNode&)
{
}

void CheckASTVisitor::postVisit(VarAssignNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncStartNode&)
{
}

void CheckASTVisitor::postVisit(FuncAssignNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::postVisit(ReturnStatementNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallStatementNode&)
{
}

void CheckASTVisitor::postVisit(LoopNode&)
{
}

void CheckASTVisitor::postVisit(ArrayNode&)
{
}

void CheckASTVisitor::postVisit(RandomAccessNode&)
{
}

void CheckASTVisitor::visit(StencilAssignmentNode& node)
{
}

void CheckASTVisitor::midVisit(StencilAssignmentNode& node)
{
}

void CheckASTVisitor::postVisit(StencilAssignmentNode& node)
{
}

void CheckASTVisitor::visit(StencilNode& node)
{
}

void CheckASTVisitor::postVisit(StencilNode& node)
{
}
