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

void CheckASTVisitor::midVisit(SequenceNode&)
{
}

void CheckASTVisitor::postVisit(SequenceNode&)
{
}

void CheckASTVisitor::visit(NumberNode&)
{
}

void CheckASTVisitor::visit(QuantityNode&)
{
}

void CheckASTVisitor::postVisit(QuantityNode&)
{
}

void CheckASTVisitor::visit(UnitNode&)
{
}

void CheckASTVisitor::visit(StringNode&)
{
}

void CheckASTVisitor::visit(TypeNode&)
{
}

void CheckASTVisitor::visit(FuncTypeNode&)
{
}

void CheckASTVisitor::visit(BinaryOpNode&)
{
}

void CheckASTVisitor::midVisit(BinaryOpNode&)
{
}

void CheckASTVisitor::postVisit(BinaryOpNode&)
{
}

void CheckASTVisitor::visit(ComparisonOpNode&)
{
}

void CheckASTVisitor::midVisit(ComparisonOpNode&)
{
}

void CheckASTVisitor::postVisit(ComparisonOpNode&)
{
}

void CheckASTVisitor::visit(NormNode&)
{
}

void CheckASTVisitor::postVisit(NormNode&)
{
}

void CheckASTVisitor::visit(UnaryNegationNode&)
{
}

void CheckASTVisitor::postVisit(UnaryNegationNode&)
{
}

void CheckASTVisitor::visit(OnNode&)
{
}

void CheckASTVisitor::midVisit(OnNode&)
{
}

void CheckASTVisitor::postVisit(OnNode&)
{
}

void CheckASTVisitor::visit(TrinaryIfNode&)
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

void CheckASTVisitor::visit(VarDeclNode&)
{
}

void CheckASTVisitor::postVisit(VarDeclNode&)
{
}

void CheckASTVisitor::visit(VarAssignNode&)
{
}

void CheckASTVisitor::postVisit(VarAssignNode&)
{
}

void CheckASTVisitor::visit(VarNode&)
{
}

void CheckASTVisitor::visit(FuncRefNode&)
{
}

void CheckASTVisitor::visit(JustAnIdentifierNode&)
{
}

void CheckASTVisitor::visit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::visit(FuncDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncDeclNode&)
{
}

void CheckASTVisitor::visit(FuncStartNode&)
{
}

void CheckASTVisitor::postVisit(FuncStartNode&)
{
}

void CheckASTVisitor::visit(FuncAssignNode&)
{
}

void CheckASTVisitor::postVisit(FuncAssignNode&)
{
}

void CheckASTVisitor::visit(FuncArgsNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::visit(ReturnStatementNode&)
{
}

void CheckASTVisitor::postVisit(ReturnStatementNode&)
{
}

void CheckASTVisitor::visit(FuncCallNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallNode&)
{
}

void CheckASTVisitor::visit(FuncCallStatementNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallStatementNode&)
{
}

void CheckASTVisitor::visit(LoopNode&)
{
}

void CheckASTVisitor::postVisit(LoopNode&)
{
}

void CheckASTVisitor::visit(ArrayNode&)
{
}

void CheckASTVisitor::postVisit(ArrayNode&)
{
}

void CheckASTVisitor::visit(RandomAccessNode&)
{
}

void CheckASTVisitor::postVisit(RandomAccessNode&)
{
}

void CheckASTVisitor::visit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::midVisit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::postVisit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::visit(StencilNode&)
{
}

void CheckASTVisitor::postVisit(StencilNode&)
{
}
