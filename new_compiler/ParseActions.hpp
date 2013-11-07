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

Node* handleNumber(const double num);

Node* handleIdentifier(const std::string& name);

VarDeclNode* handleDeclaration(const std::string& name, TypeNode* type);

VarAssignNode* handleAssignment(const std::string& name, Node* expr);

Node* handleFuncDeclaration(const std::string& name, FuncTypeNode* ftype);

Node* handleFuncStart(const std::string& name, Node* funcargs);

void handleFuncStartType();

SequenceNode* handleBlock(SequenceNode* fbody);

FuncAssignNode* handleFuncAssignment(Node* funcstart, SequenceNode* fbody);

ReturnStatementNode* handleReturnStatement(Node* expr);

Node* handleDeclarationAssign(const std::string& name, TypeNode* type, Node* expr);

TypeNode* handleCollection(TypeNode* btype, Node* gridmapping, Node* subsetof);

FuncTypeNode* handleFuncType(FuncArgsDeclNode* argtypes, TypeNode* rtype);

FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args);

FuncCallStatementNode* handleFuncCallStatement(FuncCallNode* fcall);

BinaryOpNode* handleBinaryOp(BinaryOp op, Node* left, Node* right);

ComparisonOpNode* handleComparison(ComparisonOp op, Node* left, Node* right);

NormNode* handleNorm(Node* expr_to_norm);

UnaryNegationNode* handleUnaryNegation(Node* expr_to_negate);

TrinaryIfNode* handleTrinaryIf(Node* predicate, Node* iftrue, Node* iffalse);

OnNode* handleOn(Node* left, Node* right);

OnNode* handleExtend(Node* left, Node* right);

StringNode* handleString(const std::string& content);

TypeNode* handleMutableType(TypeNode* type_expr);

TypeNode* handleSequence(TypeNode* basic_type);

TypeNode* handleArrayType(const int array_size, TypeNode* type_expr);

ArrayNode* handleArray(FuncArgsNode* expr_list);

LoopNode* handleLoopStart(const std::string& loop_variable, const std::string& loop_set);

LoopNode* handleLoopStatement(LoopNode* loop_start, SequenceNode* loop_block);

RandomAccessNode* handleRandomAccess(Node* expr, const int index);

#endif // PARSEACTIONS_HEADER_INCLUDED
