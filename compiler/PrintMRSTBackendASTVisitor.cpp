/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "PrintMRSTBackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <cctype>
#include <sstream>
#include <stdexcept>


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
    // Translate to single quoted strings.
    std::cout << '\'' << node.content().substr(1, node.content().size() - 2) << '\'';
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

void PrintMRSTBackendASTVisitor::visit(MultiplyAddNode& node)
{
}

void PrintMRSTBackendASTVisitor::midVisit(MultiplyAddNode& node)
{
}

void PrintMRSTBackendASTVisitor::postVisit(MultiplyAddNode& node)
{
}

void PrintMRSTBackendASTVisitor::visit(MultiplyDivideNode& node)
{
}

void PrintMRSTBackendASTVisitor::midVisit(MultiplyDivideNode& node)
{
}

void PrintMRSTBackendASTVisitor::postVisit(MultiplyDivideNode& node)
{
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
    if (node.leftType().isCollection()) {
        const std::string esname = SymbolTable::entitySetName(node.leftType().gridMapping());
        // Now esname can be either a user-created named set or an Equelle built-in
        // function call such as AllCells(). If the second, we must transform to
        // proper call syntax for the C++ backend.
        const char first = esname[0];
        const std::string mterm = std::isupper(first) ?
            std::string("er.") + esname
            : esname;
        std::cout << mterm;
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

void PrintMRSTBackendASTVisitor::visit(JustAnIdentifierNode& node)
{
    std::cout << node.name();
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
    std::cout << indent() << "function Res = " << node.name();
}

void PrintMRSTBackendASTVisitor::postVisit(FuncStartNode&)
{
    endl();
    ++indent_;
}

void PrintMRSTBackendASTVisitor::visit(FuncAssignNode&)
{
}

void PrintMRSTBackendASTVisitor::postVisit(FuncAssignNode&)
{
    --indent_;
    std::cout << indent() << "end";
    endl();
}

void PrintMRSTBackendASTVisitor::visit(FuncArgsNode&)
{
    std::cout << '(';
}

void PrintMRSTBackendASTVisitor::midVisit(FuncArgsNode&)
{
    std::cout << ", ";
}

void PrintMRSTBackendASTVisitor::postVisit(FuncArgsNode&)
{
    std::cout << ')';
}

void PrintMRSTBackendASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << indent() << "Res = ";
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
    std::string mname;
    if (std::isupper(first)) {
        mname += std::string("er.") + fname;
    } else {
        mname += fname;
    }
    std::cout << mname;
}

void PrintMRSTBackendASTVisitor::postVisit(FuncCallNode&)
{
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
    std::cout << indent() << "for " << node.loopVariable() << " = " << node.loopSet();
    ++indent_;
    endl();
}

void PrintMRSTBackendASTVisitor::postVisit(LoopNode&)
{
    --indent_;
    std::cout << indent() << "end";
    endl();
}

void PrintMRSTBackendASTVisitor::visit(ArrayNode&)
{
    std::cout << '{';
}

void PrintMRSTBackendASTVisitor::postVisit(ArrayNode&)
{
    std::cout << '}';
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

void PrintMRSTBackendASTVisitor::visit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintMRSTBackendASTVisitor::midVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintMRSTBackendASTVisitor::postVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintMRSTBackendASTVisitor::visit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintMRSTBackendASTVisitor::postVisit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
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
"function output = equelleProgram(er, input)\n"
"% This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"    if nargin > 1\n"
"        er.setInput(input);\n"
"    end\n"
"\n"
"    % ============= Generated code starts here ================\n";
    }

    const char* endString()
    {
        return "\n"
"    % ============= Generated code ends here ================\n"
"\n"
"    er.setInput([]);\n"
"end\n";
    }
}
