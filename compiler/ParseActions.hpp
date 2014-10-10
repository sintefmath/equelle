/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSEACTIONS_HEADER_INCLUDED
#define PARSEACTIONS_HEADER_INCLUDED

#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"
#include "ASTNodes.hpp"

// ------ Parsing event handlers ------

SequenceNode* handleProgram(SequenceNode* lineblocknode);

NumberNode* handleNumber(const double num);

QuantityNode* handleQuantity(NumberNode* number, UnitNode* unit);

UnitNode* handleUnit(const std::string& name);

UnitNode* handleUnitOp(BinaryOp op, UnitNode* left, UnitNode* right);

UnitNode* handleUnitPower(UnitNode* unit, const double num);

ExpressionNode* handleIdentifier(const std::string& name);

VarDeclNode* handleDeclaration(const std::string& name, TypeNode* type);

VarAssignNode* handleAssignment(const std::string& name, ExpressionNode* expr);

Node* handleFuncDeclaration(const std::string& name, FuncTypeNode* ftype);

void handleFuncStartType();

FuncStartNode* handleFuncAssignmentStart(const std::string& name, FuncArgsNode* args);

FuncAssignNode* handleFuncAssignment(FuncStartNode* funcstart, SequenceNode* fbody);

ReturnStatementNode* handleReturnStatement(ExpressionNode* expr);

Node* handleDeclarationAssign(const std::string& name, TypeNode* type, ExpressionNode* expr);

CollectionTypeNode* handleCollection(TypeNode* btype, ExpressionNode* gridmapping, ExpressionNode* subsetof);

TypeNode* handleStencilCollection(TypeNode* type);

FuncTypeNode* handleFuncType(FuncArgsDeclNode* argtypes, TypeNode* rtype);

FuncCallLikeNode* handleFuncCallLike(const std::string& name, FuncArgsNode* args);

FuncCallStatementNode* handleFuncCallStatement(FuncCallLikeNode* fcall);

BinaryOpNode* handleBinaryOp(BinaryOp op, ExpressionNode* left, ExpressionNode* right);

ComparisonOpNode* handleComparison(ComparisonOp op, ExpressionNode* left, ExpressionNode* right);

NormNode* handleNorm(ExpressionNode* expr_to_norm);

UnaryNegationNode* handleUnaryNegation(ExpressionNode* expr_to_negate);

TrinaryIfNode* handleTrinaryIf(ExpressionNode* predicate, ExpressionNode* iftrue, ExpressionNode* iffalse);

OnNode* handleOn(ExpressionNode* left, ExpressionNode* right);

OnNode* handleExtend(ExpressionNode* left, ExpressionNode* right);

StringNode* handleString(const std::string& content);

TypeNode* handleMutableType(TypeNode* type_expr);

TypeNode* handleSequence(TypeNode* basic_type);

TypeNode* handleArrayType(const int array_size, TypeNode* type_expr);

ArrayNode* handleArray(FuncArgsNode* expr_list);

LoopNode* handleLoopStart(const std::string& loop_variable, const std::string& loop_set);

LoopNode* handleLoopStatement(LoopNode* loop_start, SequenceNode* loop_block);

RandomAccessNode* handleRandomAccess(ExpressionNode* expr, const int index);

SequenceNode* handleStencilAssignment(FuncCallLikeNode* lhs, ExpressionNode* rhs);

StencilNode* handleStencilAccess(const std::string& name, FuncArgsNode* args);




#endif // PARSEACTIONS_HEADER_INCLUDED
