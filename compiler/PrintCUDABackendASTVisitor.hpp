/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED
#define PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED

#include "ASTVisitorInterface.hpp"
#include "EquelleType.hpp"
#include <string>
#include <set>

class PrintCUDABackendASTVisitor : public ASTVisitorInterface
{
public:
    PrintCUDABackendASTVisitor();
    ~PrintCUDABackendASTVisitor();

    void visit(SequenceNode& node);
    void midVisit(SequenceNode& node);
    void postVisit(SequenceNode& node);
    void visit(NumberNode& node);
    void visit(StringNode& node);
    void visit(TypeNode& node);
    void visit(FuncTypeNode& node);
    void visit(BinaryOpNode& node);
    void midVisit(BinaryOpNode& node);
    void postVisit(BinaryOpNode& node);
    void visit(ComparisonOpNode& node);
    void midVisit(ComparisonOpNode& node);
    void postVisit(ComparisonOpNode& node);
    void visit(NormNode& node);
    void postVisit(NormNode& node);
    void visit(UnaryNegationNode& node);
    void postVisit(UnaryNegationNode& node);
    void visit(OnNode& node);
    void midVisit(OnNode& node);
    void postVisit(OnNode& node);
    void visit(TrinaryIfNode& node);
    void questionMarkVisit(TrinaryIfNode& node);
    void colonVisit(TrinaryIfNode& node);
    void postVisit(TrinaryIfNode& node);
    void visit(VarDeclNode& node);
    void postVisit(VarDeclNode& node);
    void visit(VarAssignNode& node);
    void postVisit(VarAssignNode& node);
    void visit(VarNode& node);
    void visit(FuncRefNode& node);
    void visit(JustAnIdentifierNode& node);
    void visit(FuncArgsDeclNode& node);
    void midVisit(FuncArgsDeclNode& node);
    void postVisit(FuncArgsDeclNode& node);
    void visit(FuncDeclNode& node);
    void postVisit(FuncDeclNode& node);
    void visit(FuncStartNode& node);
    void postVisit(FuncStartNode& node);
    void visit(FuncAssignNode& node);
    void postVisit(FuncAssignNode& node);
    void visit(FuncArgsNode& node);
    void midVisit(FuncArgsNode& node);
    void postVisit(FuncArgsNode& node);
    void visit(ReturnStatementNode& node);
    void postVisit(ReturnStatementNode& node);
    void visit(FuncCallLikeNode& node);
    void postVisit(FuncCallLikeNode& node);
    void visit(FuncCallStatementNode& node);
    void postVisit(FuncCallStatementNode& node);
    void visit(LoopNode& node);
    void postVisit(LoopNode& node);
    void visit(ArrayNode& node);
    void postVisit(ArrayNode& node);
    void visit(RandomAccessNode& node);
    void postVisit(RandomAccessNode& node);
    void visit(StencilAssignmentNode& node);
    void midVisit(StencilAssignmentNode& node);
    void postVisit(StencilAssignmentNode& node);

    // These are overriden by subclasses who only need to alter the surroundings of the generated code.
    virtual const char* cppStartString() const;
    virtual const char* cppEndString() const;

private:
    bool suppressed_;
    int indent_;
    int sequence_depth_;
    std::set<std::string> requirement_strings_;
    void endl() const;
    std::string indent() const;
    void suppress();
    void unsuppress();
    std::string cppTypeString(const EquelleType& et) const;
    void addRequirementString(const std::string& req);
};

#endif // PRINTCUDABACKENDASTVISITOR_HEADER_INCLUDED
