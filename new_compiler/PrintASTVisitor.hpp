/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PRINTASTVISITOR_HEADER_INCLUDED
#define PRINTASTVISITOR_HEADER_INCLUDED

#include "ASTVisitorInterface.hpp"
#include <string>


class PrintASTVisitor : public ASTVisitorInterface
{
public:
    PrintASTVisitor();
    ~PrintASTVisitor();

    void visit(SequenceNode& node);
    void visit(NumberNode& node);
    void visit(StringNode& node);
    void visit(TypeNode& node);
    void visit(FuncTypeNode& node);
    void visit(BinaryOpNode& node);
    void visit(NormNode& node);
    void visit(UnaryNegationNode& node);
    void visit(OnNode& node);
    void visit(TrinaryIfNode& node);
    void visit(VarDeclNode& node);
    void visit(VarAssignNode& node);
    void visit(VarNode& node);
    void visit(FuncRefNode& node);
    void visit(JustAnIdentifierNode& node);
    void visit(FuncArgsDeclNode& node);
    void visit(FuncDeclNode& node);
    void visit(FuncStartNode& node);
    void visit(FuncAssignNode& node);
    void visit(FuncArgsNode& node);
    void visit(ReturnStatementNode& node);
    void visit(FuncCallNode& node);
    void visit(FuncCallStatementNode& node);

    void midVisit(SequenceNode& node);
    void postVisit(SequenceNode& node);
    void midVisit(BinaryOpNode& node);
    void postVisit(BinaryOpNode& node);
    void postVisit(NormNode& node);
    void postVisit(UnaryNegationNode& node);
    void midVisit(OnNode& node);
    void postVisit(OnNode& node);
    void questionMarkVisit(TrinaryIfNode& node);
    void colonVisit(TrinaryIfNode& node);
    void postVisit(TrinaryIfNode& node);
    void postVisit(VarDeclNode& node);
    void postVisit(VarAssignNode& node);
    void midVisit(FuncArgsDeclNode& node);
    void postVisit(FuncArgsDeclNode& node);
    void postVisit(FuncDeclNode& node);
    void postVisit(FuncStartNode& node);
    void postVisit(FuncAssignNode& node);
    void midVisit(FuncArgsNode& node);
    void postVisit(FuncArgsNode& node);
    void postVisit(ReturnStatementNode& node);
    void postVisit(FuncCallNode& node);
    void postVisit(FuncCallStatementNode& node);

private:
    int indent_;
    std::string indent() const;
};


#endif // PRINTASTVISITOR_HEADER_INCLUDED
