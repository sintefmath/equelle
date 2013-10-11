/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PRINTEQUELLEASTVISITOR_HEADER_INCLUDED
#define PRINTEQUELLEASTVISITOR_HEADER_INCLUDED

#include "ASTVisitorInterface.hpp"
#include <string>

class PrintEquelleASTVisitor : public ASTVisitorInterface
{
public:
    PrintEquelleASTVisitor();
    ~PrintEquelleASTVisitor();

    void visit(SequenceNode& node);
    void postVisit(SequenceNode& node);
    void visit(NumberNode& node);
    void visit(TypeNode& node);
    void visit(FuncTypeNode& node);
    void visit(BinaryOpNode& node);
    void postVisit(BinaryOpNode& node);
    void visit(NormNode& node);
    void postVisit(NormNode& node);
    void visit(UnaryNegationNode& node);
    void postVisit(UnaryNegationNode& node);
    void visit(OnNode& node);
    void postVisit(OnNode& node);
    void visit(TrinaryIfNode& node);
    void postVisit(TrinaryIfNode& node);
    void visit(VarDeclNode& node);
    void postVisit(VarDeclNode& node);
    void visit(VarAssignNode& node);
    void postVisit(VarAssignNode& node);
    void visit(VarNode& node);
    void visit(FuncRefNode& node);
    void visit(JustAnIdentifierNode& node);
    void visit(FuncArgsDeclNode& node);
    void postVisit(FuncArgsDeclNode& node);
    void visit(FuncDeclNode& node);
    void postVisit(FuncDeclNode& node);
    void visit(FuncStartNode& node);
    void postVisit(FuncStartNode& node);
    void visit(FuncAssignNode& node);
    void postVisit(FuncAssignNode& node);
    void visit(FuncArgsNode& node);
    void postVisit(FuncArgsNode& node);
    void visit(ReturnStatementNode& node);
    void postVisit(ReturnStatementNode& node);
    void visit(FuncCallNode& node);
    void postVisit(FuncCallNode& node);

private:
    int indent_;
    void endl() const;
    std::string indent() const;
};

#endif // PRINTEQUELLEASTVISITOR_HEADER_INCLUDED
