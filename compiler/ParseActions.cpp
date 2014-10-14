/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "Common.hpp"
#include "EquelleType.hpp"
#include "ASTNodes.hpp"
#include "ParseActions.hpp"
#include "Dimension.hpp"
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <cmath>



// ------ Parsing event handlers ------


SequenceNode* handleProgram(SequenceNode* lineblocknode)
{
    SymbolTable::setProgram(lineblocknode);
    return lineblocknode;
}



NumberNode* handleNumber(const double num)
{
    NumberNode* node = new NumberNode(num);
    node->setLocation(FileLocation(yylineno));
    return node;
}



QuantityNode* handleQuantity(NumberNode* number, UnitNode* unit)
{
    QuantityNode* node = new QuantityNode(number, unit);
    node->setLocation(FileLocation(yylineno));
    return node;
}



UnitNode* handleUnit(const std::string& name)
{
    UnitNode* node = new UnitNode(name);
    node->setLocation(FileLocation(yylineno));
    return node;
}



UnitNode* handleUnitOp(BinaryOp op, UnitNode* left, UnitNode* right)
{
    Dimension d = left->dimension();
    double c = left->conversionFactorSI();
    switch (op) {
    case Multiply:
        d = d + right->dimension();
        c = c * right->conversionFactorSI();
        break;
    case Divide:
        d = d - right->dimension();
        c = c / right->conversionFactorSI();
        break;
    default:
        yyerror("Units can only be manipulated with '*', '/' or '^'.");
    }
    delete left;
    delete right;
    UnitNode* node = new UnitNode(d, c);
    node->setLocation(FileLocation(yylineno));
    return node;
}



UnitNode* handleUnitPower(UnitNode* unit, const double num)
{
    const int n = static_cast<int>(num);
    if (n != num) {
        yyerror("Powers of units (to the right of '^') can only be integers.");
    }
    const Dimension d = unit->dimension() * n;
    const double c = std::pow(unit->conversionFactorSI(), n);
    delete unit;
    UnitNode* node = new UnitNode(d, c);
    node->setLocation(FileLocation(yylineno));
    return node;
}



ExpressionNode* handleIdentifier(const std::string& name)
{
    VarNode* node = new VarNode(name);
    node->setLocation(FileLocation(yylineno));
    return node;
#if 0
    if (SymbolTable::isVariableDeclared(name)) {
        return new VarNode(name);
    } else {
        if (SymbolTable::isFunctionDeclared(name)) {
            return new FuncRefNode(name);
        } else {
            // This is a small problem: we want the error below, to catch
            // usage of undeclared identifiers, but the function start section
            // would then generate errors, because we are not yet in the function
            // scope.

            // std::string e("unknown identifier ");
            // e += name;
            // yyerror(e.c_str());
            return new JustAnIdentifierNode(name);
        }
    }
#endif
}



VarDeclNode* handleDeclaration(const std::string& name, TypeNode* type)
{
    VarDeclNode* node = new VarDeclNode(name, type);
    node->setLocation(FileLocation(yylineno));
    return node;
}



VarAssignNode* handleAssignment(const std::string& name, ExpressionNode* expr)
{
    VarAssignNode* node = new VarAssignNode(name, expr);
    node->setLocation(FileLocation(yylineno));
    return node;
}



Node* handleFuncDeclaration(const std::string& name, FuncTypeNode* ftype)
{
    FuncDeclNode* node = new FuncDeclNode(name, ftype);
    node->setLocation(FileLocation(yylineno));
    return node;
}



void handleFuncStartType()
{
}


FuncStartNode* handleFuncAssignmentStart(const std::string& name, FuncArgsNode* args)
{
    FuncStartNode* node = new FuncStartNode(name, args);
    node->setLocation(FileLocation(yylineno));
    return node;
#if 0
    // We are dealing with a function
    if (SymbolTable::isFunctionDeclared(name)) {
        // Set the scope name for the following block (the function itself)
        // Will be "undone" in handleFuncAssignment
        SymbolTable::setCurrentFunction(name);
        return new FuncStartNode(name, args);
    }
    // We are dealing with a defined stencil variable
    else if (SymbolTable::isVariableDeclared(name)) {
        return handleStencilAccess(name, args);
    }
    // We are dealing with an undefined stencil variable
    else {
        EquelleType type(Invalid, Collection);
        type.setStencil(true);
        SymbolTable::declareVariable(name, type);
        return handleStencilAccess(name, args);
    }
#endif
}


FuncAssignNode* handleFuncAssignment(FuncStartNode* funcstart, SequenceNode* fbody)
{
    FuncAssignNode* node = new FuncAssignNode(funcstart, fbody);
    node->setLocation(FileLocation(yylineno));
    return node;
}



ReturnStatementNode* handleReturnStatement(ExpressionNode* expr)
{
    ReturnStatementNode* node = new ReturnStatementNode(expr);
    node->setLocation(FileLocation(yylineno));
    return node;
}



Node* handleDeclarationAssign(const std::string& name, TypeNode* type, ExpressionNode* expr)
{
    SequenceNode* seq = new SequenceNode;
    seq->pushNode(handleDeclaration(name, type));
    seq->pushNode(handleAssignment(name, expr));
    seq->setLocation(FileLocation(yylineno));
    return seq;
}



CollectionTypeNode* handleCollection(TypeNode* btype, ExpressionNode* gridmapping, ExpressionNode* subsetof)
{
    CollectionTypeNode* node = new CollectionTypeNode(btype, gridmapping, subsetof);
    node->setLocation(FileLocation(yylineno));
    return node;
}



TypeNode* handleStencilCollection(TypeNode* type_expr)
{
    EquelleType et = type_expr->type();
    et.setStencil(true);
    TypeNode* tn = new TypeNode(et);
    tn->setLocation(type_expr->location());
    delete type_expr;
    return tn;
}



FuncTypeNode* handleFuncType(FuncArgsDeclNode* argtypes, TypeNode* rtype)
{
    FuncTypeNode* node = new FuncTypeNode(argtypes, rtype);
    node->setLocation(FileLocation(yylineno));
    return node;
}



StencilNode* handleStencilAccess(const std::string& name, FuncArgsNode* args)
{
    StencilNode* node = new StencilNode(name, args);
    node->setLocation(FileLocation(yylineno));
    return node;
#if 0
    if (!SymbolTable::isVariableDeclared(name)) {
        std::string err_msg = "Could not find the stencil variable " + name;
        yyerror(err_msg.c_str());
    }

    auto argtypes = args->argumentTypes();
    for (int i=0; i<argtypes.size(); ++i) {
        if (!isStencilType(argtypes[i].basicType())) {
            std::stringstream err_msg;
            err_msg << "Cannot access a stencil with a non-stencil index in variable \""
                    << name << "\"" << std::endl;
            yyerror(err_msg.str().c_str());
        }
        else if (argtypes[i].basicType() != StencilI + i) {
            std::stringstream err_msg;
            err_msg << "Got index " << basicTypeString(argtypes[i].basicType())
                    << " but expected " << basicTypeString(BasicType(StencilI + i))
                    << " for variable \"" << name << "\"" << std::endl;
            yyerror(err_msg.str().c_str());
        }
    }
    return new StencilNode(name, args);
#endif
}

FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args)
{
    FuncCallNode* node = new FuncCallNode(name, args);
    node->setLocation(FileLocation(yylineno));
    return node;
}

FuncCallLikeNode* handleFuncCallLike(const std::string& name, FuncArgsNode* args)
{
    return handleFuncCall(name, args);
#if 0
    if (SymbolTable::isFunctionDeclared(name)) {
        return handleFuncCall(name, args);
    }
    else if (SymbolTable::isVariableDeclared(name)) {
        return handleStencilAccess(name, args);
    }
    else {
        std::string err_msg = "Could not find the stencil variable or function " + name;
        yyerror(err_msg.c_str());
        return nullptr;
    }
#endif
}

FuncCallStatementNode* handleFuncCallStatement(FuncCallLikeNode* fcall_like)
{
    FuncCallNode* fcall = dynamic_cast<FuncCallNode*>(fcall_like);
    if (fcall == nullptr) {
        std::string err_msg = "Internal error: The function \"" + fcall_like->name() + "\" does not appear to be properly defined";
        yyerror(err_msg.c_str());
    }
    FuncCallStatementNode* node = new FuncCallStatementNode(fcall);
    node->setLocation(FileLocation(yylineno));
    return node;
}



BinaryOpNode* handleBinaryOp(BinaryOp op, ExpressionNode* left, ExpressionNode* right)
{
    BinaryOpNode* node = new BinaryOpNode(op, left, right);
    node->setLocation(FileLocation(yylineno));
    return node;
}



ComparisonOpNode* handleComparison(ComparisonOp op, ExpressionNode* left, ExpressionNode* right)
{
    ComparisonOpNode* node = new ComparisonOpNode(op, left, right);
    node->setLocation(FileLocation(yylineno));
    return node;
}



NormNode* handleNorm(ExpressionNode* expr_to_norm)
{
    NormNode* node = new NormNode(expr_to_norm);
    node->setLocation(FileLocation(yylineno));
    return node;
}



UnaryNegationNode* handleUnaryNegation(ExpressionNode* expr_to_negate)
{
    UnaryNegationNode* node = new UnaryNegationNode(expr_to_negate);
    node->setLocation(FileLocation(yylineno));
    return node;
}



TrinaryIfNode* handleTrinaryIf(ExpressionNode* predicate, ExpressionNode* iftrue, ExpressionNode* iffalse)
{
    TrinaryIfNode* node = new TrinaryIfNode(predicate, iftrue, iffalse);
    node->setLocation(FileLocation(yylineno));
    return node;
}



OnNode* handleOn(ExpressionNode* left, ExpressionNode* right)
{
    OnNode* node = new OnNode(left, right, false);
    node->setLocation(FileLocation(yylineno));
    return node;
}



OnNode* handleExtend(ExpressionNode* left, ExpressionNode* right)
{
    OnNode* node = new OnNode(left, right, true);
    node->setLocation(FileLocation(yylineno));
    return node;
}



StringNode* handleString(const std::string& content)
{
    StringNode* node = new StringNode(content);
    node->setLocation(FileLocation(yylineno));
    return node;
}



TypeNode* handleMutableType(TypeNode* type_expr)
{
    MutableTypeNode* tn = new MutableTypeNode(type_expr);
    tn->setLocation(FileLocation(yylineno));
    return tn;
#if 0
    EquelleType et = type_expr->type();
    et.setMutable(true);
    TypeNode* tn = new TypeNode(et);
    tn->setLocation(type_expr->location());
    delete type_expr;
    return tn;
#endif
}



TypeNode* handleSequenceType(TypeNode* type_expr)
{
    SequenceTypeNode* tn = new SequenceTypeNode(type_expr);
    tn->setLocation(FileLocation(yylineno));
    return tn;
#if 0
    const EquelleType et = type_expr->type();
    if (!et.isBasic()) {
        yyerror("cannot create a Sequence of non-basic types.");
    }
    TypeNode* node = new TypeNode(EquelleType(et.basicType(), Sequence, et.gridMapping(), et.subsetOf()));
    node->setLocation(FileLocation(yylineno));
    return node;
#endif
}



TypeNode* handleArrayType(const int array_size, TypeNode* type_expr)
{
    ArrayTypeNode* tn = new ArrayTypeNode(type_expr, array_size);
    tn->setLocation(FileLocation(yylineno));
    return tn;
#if 0
    EquelleType et = type_expr->type();
    if (et.isArray()) {
        yyerror("cannot create an Array of an Array.");
        return type_expr;
    } else {
        et.setArraySize(array_size);
        TypeNode* tn = new TypeNode(et);
        tn->setLocation(type_expr->location());
        delete type_expr;
        return tn;
    }
#endif
}



ArrayNode* handleArray(FuncArgsNode* expr_list)
{
    ArrayNode* node = new ArrayNode(expr_list);
    node->setLocation(FileLocation(yylineno));
    return node;
}



LoopNode* handleLoopStart(const std::string& loop_variable, const std::string& loop_set)
{
    LoopNode* node = new LoopNode(loop_variable, loop_set);
    node->setLocation(FileLocation(yylineno));
    return node;
}



LoopNode* handleLoopStatement(LoopNode* loop_start, SequenceNode* loop_block)
{
    loop_start->setBlock(loop_block);
    return loop_start;
}



RandomAccessNode* handleRandomAccess(ExpressionNode* expr, const int index)
{
    RandomAccessNode* node = new RandomAccessNode(expr, index);
    node->setLocation(FileLocation(yylineno));
    return node;
}



SequenceNode* handleStencilAssignment(FuncCallLikeNode* lhs, ExpressionNode* rhs)
{
    SequenceNode* retval = new SequenceNode();
    retval->setLocation(FileLocation(yylineno));

    StencilNode* stencil = dynamic_cast<StencilNode*>(lhs);
    if (stencil == nullptr) {
        std::string err_msg = "Internal error: The stencil \"" + lhs->name() + "\" does not appear to be properly defined";
        yyerror(err_msg.c_str());
    }

#if 0
    // If the type is a collection of invalids (no pun intended)
    // we can safely set it to the type of the rhs
    EquelleType lhs_et = SymbolTable::variableType(stencil->name());
    if (lhs_et.basicType() == Invalid
        && lhs_et.compositeType() == Collection
        && lhs_et.isStencil()) {
        EquelleType lhs_et = rhs->type();
        lhs_et.setMutable(true);
        SymbolTable::setVariableType(stencil->name(), lhs_et);
        // TODO: set dimensions correctly here
        TypeNode* lhs_type = new TypeNode(lhs_et);
        retval->pushNode(new VarDeclNode(stencil->name(), lhs_type));
    }
#endif
    retval->pushNode(new StencilAssignmentNode(stencil, rhs));
    return retval;
}

