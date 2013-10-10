/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSER_HEADER_INCLUDED
#define PARSER_HEADER_INCLUDED

#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"
#include "ASTNodes.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <algorithm>
#include <stdexcept>
#include <cassert>





// ------ Handle parse events ------

inline NodePtr handleNumber(const double num)
{
    return new NumberNode(num);
}

inline VarDeclNode* handleDeclaration(const std::string name, TypeNodePtr type)
{
    SymbolTable::declareVariable(name, type->type());
    return new VarDeclNode(name, type);
}

inline VarAssignNode* handleAssignment(const std::string name, NodePtr expr)
{
    // If already declared...
    if (SymbolTable::isVariableDeclared(name)) {
        // Check if already assigned.
        if (SymbolTable::isVariableAssigned(name)) {
            std::string err_msg = "variable already assigned, cannot re-assign ";
            err_msg += name;
            yyerror(err_msg.c_str());
            return nullptr;
        }
        // Check that declared type matches right hand side.
        EquelleType lhs_type = SymbolTable::variableType(name);
        EquelleType rhs_type = expr->type();
        if (lhs_type != rhs_type) {
            // Check for special case: variable declared to have type
            // 'Collection Of <Entity> Subset Of <Some entityset>',
            // actual setting its grid mapping is postponed until the entityset
            // has been created by a function call.
            // That means we have to set the actual grid mapping here.
            if (lhs_type.gridMapping() == PostponedDefinition
                && lhs_type.basicType() == rhs_type.basicType()
                && lhs_type.isCollection() && rhs_type.isCollection()
                && SymbolTable::isSubset(rhs_type.gridMapping(), lhs_type.subsetOf())) {
                // OK, should make postponed definition of the variable.
                SymbolTable::setVariableType(name, rhs_type);
            } else {
                std::string err_msg = "mismatch between type in assignment and declaration for ";
                err_msg += name;
                yyerror(err_msg.c_str());
                return nullptr;
            }
        }
    } else {
        SymbolTable::declareVariable(name, expr->type());
    }

    // Set variable to assigned and return.
    SymbolTable::setVariableAssigned(name, true);
    return new VarAssignNode(name, expr);
}

inline NodePtr handleFuncDeclaration(const std::string name, FuncTypeNodePtr ftype)
{
    SymbolTable::declareFunction(name, ftype->funcType());
    return new FuncDeclNode(name, ftype);
}

inline NodePtr handleDeclarationAssign(const std::string name, TypeNodePtr type, NodePtr expr)
{
    SequenceNode* seq = new SequenceNode;
    seq->pushNode(handleDeclaration(name, type));
    seq->pushNode(handleAssignment(name, expr));
    return seq;
}

inline TypeNodePtr handleCollection(TypeNodePtr btype, NodePtr gridmapping, NodePtr subsetof)
{
    assert(gridmapping == nullptr || subsetof == nullptr);
    EquelleType bt = btype->type();
    if (!bt.isBasic()) {
        std::string errmsg = "attempting to declare a Collection Of <nonsense>";
        yyerror(errmsg.c_str());
    }
    int gm = NotApplicable;
    if (gridmapping) {
        if (!gridmapping->type().isEntityCollection() || gridmapping->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be On a Collection of Cell, Face etc.");
        } else {
            gm = gridmapping->type().gridMapping();
        }
    }
    int subset = NotApplicable;
    if (subsetof) {
        // We are creating a new entity collection.
        if (!subsetof->type().isEntityCollection() || subsetof->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be Subset Of a Collection of Cell, Face etc.");
        } else {
            gm = PostponedDefinition;
            subset = subsetof->type().gridMapping();
        }
    }
    return new TypeNode(EquelleType(bt.basicType(), true, gm, subset));
}

inline FuncTypeNodePtr handleFuncType(FuncArgsDeclNode* argtypes, TypeNodePtr rtype)
{
    return new FuncTypeNode(FunctionType(argtypes->arguments(), rtype->type()));
}

inline FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args)
{
    const Function& f = SymbolTable::getFunction(name);
    int dynsubret = f.functionType().dynamicSubsetReturn(args->argumentTypes());
    if (dynsubret != NotApplicable) {
        // Create a new entity collection. This is the only place this can happen.
        const int gm = SymbolTable::declareNewEntitySet(dynsubret);
        return new FuncCallNode(name, args, gm);
    } else {
        return new FuncCallNode(name, args);
    }
}

inline BinaryOpNode* handleBinaryOp(BinaryOp op, Node* left, Node* right)
{
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if (!isNumericType(lt.basicType()) || !(isNumericType(rt.basicType()))) {
        yyerror("arithmetic binary operators only apply to numeric types");
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            yyerror("arithmetic binary operators on 'Collection's only acceptable "
                    "if both sides are 'On' the same set.");
        }
    }
    switch (op) {
    case Add:
        // Intentional fall-through.
    case Subtract:
        if (lt != rt) {
            yyerror("addition and subtraction only allowed between identical types.");
        }
        break;
    case Multiply:
        if (lt.basicType() == Vector && rt.basicType() == Vector) {
            yyerror("cannot multiply two 'Vector' types.");
        }
        break;
    case Divide:
        if (rt.basicType() != Scalar) {
            yyerror("can only divide by 'Scalar' types");
        }
        break;
    default:
        yyerror("internal compiler error in handleBinaryOp().");
    }
    return new BinaryOpNode(op, left, right);
}

inline NormNode* handleNorm(NodePtr expr_to_norm)
{
    const BasicType bt = expr_to_norm->type().basicType();
    if (isEntityType(bt) || bt == Scalar || bt == Vector) {
        return new NormNode(expr_to_norm);
    } else {
        yyerror("can only take norm of Scalar, Vector, Cell, Face, Edge and Vertex types.");
        return new NormNode(expr_to_norm);
    }
}

#endif // PARSER_HEADER_INCLUDED
