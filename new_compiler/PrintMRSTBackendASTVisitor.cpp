/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "PrintMRSTBackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <cctype>
#include <sstream>


namespace
{
    const char* startString();
    const char* endString();
}



PrintMRSTBackendASTVisitor::PrintMRSTBackendASTVisitor()
    : indent_(1),
      sequence_depth_(0)
{
}

PrintMRSTBackendASTVisitor::~PrintMRSTBackendASTVisitor()
{
}





void PrintMRSTBackendASTVisitor::visit(SequenceNode&)
{
    if (sequence_depth_ == 0) {
        // This is the root node of the program.
        std::cout << startString();
        endl();
    }
    ++sequence_depth_;
}

void PrintMRSTBackendASTVisitor::midVisit(SequenceNode&)
{
}

void PrintMRSTBackendASTVisitor::postVisit(SequenceNode&)
{
    --sequence_depth_;
    if (sequence_depth_ == 0) {
        // We are back at the root node. Finish main() function.
        std::cout << endString();
    }
}

void PrintMRSTBackendASTVisitor::visit(NumberNode& node)
{
    std::cout.precision(16);
    std::cout << node.number();
}

void PrintMRSTBackendASTVisitor::visit(StringNode& node)
{
    std::cout << node.content();
}

void PrintMRSTBackendASTVisitor::visit(TypeNode&)
{
    // std::cout << SymbolTable::equelleString(node.type());
}

void PrintMRSTBackendASTVisitor::visit(FuncTypeNode&)
{
    // std::cout << node.funcType().equelleString();
}

void PrintMRSTBackendASTVisitor::visit(BinaryOpNode&)
{
    std::cout << '(';
}

void PrintMRSTBackendASTVisitor::midVisit(BinaryOpNode& node)
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

void PrintMRSTBackendASTVisitor::postVisit(BinaryOpNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(ComparisonOpNode&)
{
    std::cout << '(';
}

void PrintMRSTBackendASTVisitor::midVisit(ComparisonOpNode& node)
{
    std::string op(" ");
    switch (node.op()) {
    case Less:
        op = "<";
        break;
    case Greater:
        op = ">";
        break;
    case LessEqual:
        op = "<=";
        break;
    case GreaterEqual:
        op = ">=";
        break;
    case Equal:
        op = "==";
        break;
    case NotEqual:
        op = "!=";
        break;
    default:
        break;
    }
    std::cout << ' ' << op << ' ';
}

void PrintMRSTBackendASTVisitor::postVisit(ComparisonOpNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(NormNode&)
{
    std::cout << "eqNorm(";
}

void PrintMRSTBackendASTVisitor::postVisit(NormNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(UnaryNegationNode&)
{
    std::cout << '-';
}

void PrintMRSTBackendASTVisitor::postVisit(UnaryNegationNode&)
{
}

void PrintMRSTBackendASTVisitor::visit(OnNode& node)
{
    if (node.isExtend()) {
        std::cout << "eqOperatorExtend(";
    } else {
        std::cout << "eqOperatorOn(";
    }
}

void PrintMRSTBackendASTVisitor::midVisit(OnNode& node)
{
    // Backend's operatorOn/operatorExtend has three arguments when the left argument
    // is a collection, not two. The middle argument (that we will
    // write in this method) should be the set that the first argument
    // is On. Example:
    // a : Collection Of Scalar On InteriorFaces()
    // a On AllFaces() ===> er.operatorOn(a, InteriorFaces(), AllFaces()).
    std::cout << ", ";
    if (node.lefttype().isCollection()) {
        const std::string esname = SymbolTable::entitySetName(node.lefttype().gridMapping());
        // Now esname can be either a user-created named set or an Equelle built-in
        // function call such as AllCells(). If the second, we must transform to
        // proper call syntax for the C++ backend.
        const char first = esname[0];
        const std::string cppterm = std::isupper(first) ?
            std::string("eq") + esname
            : esname;
        std::cout << cppterm;
        std::cout << ", ";
    }
}

void PrintMRSTBackendASTVisitor::postVisit(OnNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(TrinaryIfNode&)
{
    std::cout << "eqTrinaryIf(";
}

void PrintMRSTBackendASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
    std::cout << ", ";
}

void PrintMRSTBackendASTVisitor::colonVisit(TrinaryIfNode&)
{
    std::cout << ", ";
}

void PrintMRSTBackendASTVisitor::postVisit(TrinaryIfNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(VarDeclNode&)
{
}

void PrintMRSTBackendASTVisitor::postVisit(VarDeclNode&)
{
}

void PrintMRSTBackendASTVisitor::visit(VarAssignNode& node)
{
    std::cout << indent() << node.name() << " = ";
}

void PrintMRSTBackendASTVisitor::postVisit(VarAssignNode&)
{
    std::cout << ';';
    endl();
}

void PrintMRSTBackendASTVisitor::visit(VarNode& node)
{
    std::cout << node.name();
}

void PrintMRSTBackendASTVisitor::visit(FuncRefNode& node)
{
    std::cout << node.name();
}

void PrintMRSTBackendASTVisitor::visit(JustAnIdentifierNode&)
{
}

void PrintMRSTBackendASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintMRSTBackendASTVisitor::midVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::midVisit()}";
}

void PrintMRSTBackendASTVisitor::postVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintMRSTBackendASTVisitor::visit(FuncDeclNode&)
{
    // std::cout << node.name() << " : ";
}

void PrintMRSTBackendASTVisitor::postVisit(FuncDeclNode&)
{
    // endl();
}

void PrintMRSTBackendASTVisitor::visit(FuncStartNode& node)
{
    // std::cout << indent() << "auto " << node.name() << " = [&](";
    // const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    // const size_t n = ft.arguments().size();
    // for (int i = 0; i < n; ++i) {
    //     std::cout << "const "
    //               << cppTypeString(ft.arguments()[i].type())
    //               << "& " << ft.arguments()[i].name();
    //     if (i < n - 1) {
    //         std::cout << ", ";
    //     }
    // }
    std::cout << "Function start " << node.name();
    ++indent_;
}

void PrintMRSTBackendASTVisitor::postVisit(FuncStartNode&)
{
    // const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    // std::cout << ") -> " << cppTypeString(ft.returnType()) << " {";
    endl();
}

void PrintMRSTBackendASTVisitor::visit(FuncAssignNode&)
{
}

void PrintMRSTBackendASTVisitor::postVisit(FuncAssignNode&)
{
    --indent_;
    std::cout << indent() << "};";
    endl();
}

void PrintMRSTBackendASTVisitor::visit(FuncArgsNode&)
{
}

void PrintMRSTBackendASTVisitor::midVisit(FuncArgsNode&)
{
    std::cout << ", ";
}

void PrintMRSTBackendASTVisitor::postVisit(FuncArgsNode&)
{
}

void PrintMRSTBackendASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << indent() << "return ";
}

void PrintMRSTBackendASTVisitor::postVisit(ReturnStatementNode&)
{
    std::cout << ';';
    endl();
}

void PrintMRSTBackendASTVisitor::visit(FuncCallNode& node)
{
    const std::string fname = node.name();
    const char first = fname[0];
    std::string cppname;
    if (std::isupper(first)) {
        cppname += std::string("eq") + fname;
    } else {
        cppname += fname;
    }
    std::cout << cppname << '(';
}

void PrintMRSTBackendASTVisitor::postVisit(FuncCallNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(FuncCallStatementNode&)
{
    std::cout << indent();
}

void PrintMRSTBackendASTVisitor::postVisit(FuncCallStatementNode&)
{
    std::cout << ';';
    endl();
}

void PrintMRSTBackendASTVisitor::visit(LoopNode& node)
{
    std::cout << indent() << "for " << node.loopVariable() << " = " << node.loopSet() << ")";
    ++indent_;
    endl();
}

void PrintMRSTBackendASTVisitor::postVisit(LoopNode&)
{
    --indent_;
    std::cout << indent() << "end";
    endl();
}

void PrintMRSTBackendASTVisitor::visit(RandomAccessNode&)
{
}

void PrintMRSTBackendASTVisitor::postVisit(RandomAccessNode& node)
{
    // Random access op is taking the column of the underlying matrix.
    // Also, we need to add one, since Matlab starts from 1 not 0.
    std::cout << "(:, " << node.index() + 1 << ")";
}



void PrintMRSTBackendASTVisitor::endl() const
{
    std::cout << '\n';
}

std::string PrintMRSTBackendASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}


namespace
{
    const char* startString()
    {
        return
"function output = equelleProgram(input)\n"
"% This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"    % ============= Generated code starts here ================\n";
    }

    const char* endString()
    {
        return "\n"
"    % ============= Generated code ends here ================\n"
"\n"
"endfunction\n";
    }
}
