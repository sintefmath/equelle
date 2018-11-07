/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <stdexcept>



PrintASTVisitor::PrintASTVisitor()
    : indent_(0)
{
}

PrintASTVisitor::~PrintASTVisitor()
{
}



void PrintASTVisitor::visit(SequenceNode&)
{
    if (indent_ == 0) {
        SymbolTable::dump();
    }
    std::cout << indent() << "SequenceNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(NumberNode& node)
{
    std::cout << indent() << "NumberNode: " << node.number() << '\n';
}

void PrintASTVisitor::visit(QuantityNode& node)
{
    std::cout << indent() << "QuantityNode" << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(BasicUnitNode& node)
{
    std::cout.precision(15);
    std::cout << indent() << "BasicUnitNode: (factor = " << node.conversionFactorSI() << "), (dimension = " << node.dimension() << ")\n";
}

void PrintASTVisitor::visit(BinaryOpUnitNode& node)
{
    char op = ' ';
    switch (node.op()) {
    case Multiply:
        op = '*';
        break;
    case Divide:
        op = '/';
        break;
    default:
        break;
    }
    std::cout << indent() << "BinaryOpUnitNode: " << op << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(PowerUnitNode& node)
{
    std::cout << indent() << "PowerUnitNode: power = " << node.power() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(StringNode& node)
{
    std::cout << indent() << "StringNode: " << node.content() << '\n';
}

void PrintASTVisitor::visit(TypeNode& node)
{
    std::cout << indent() << "TypeNode: " << SymbolTable::equelleString(node.type()) << '\n';
}

void PrintASTVisitor::visit(FuncTypeNode& node)
{
    std::cout << indent() << "FuncTypeNode: " << node.funcType().equelleString() << '\n';
}

void PrintASTVisitor::visit(BinaryOpNode& node)
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
    std::cout << indent() << "BinaryOpNode: " << op << '\n';

    ++indent_;
}

void PrintASTVisitor::visit(MultiplyAddNode& node)
{
}

void PrintASTVisitor::visit(MultiplyDivideNode& node)
{
}

void PrintASTVisitor::visit(ComparisonOpNode& node)
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
    std::cout << indent() << "ComparisonOpNode: " << op << '\n';

    ++indent_;
}

void PrintASTVisitor::visit(NormNode&)
{
    std::cout << indent() << "NormNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(UnaryNegationNode&)
{
    std::cout << indent() << "UnaryNegationNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(OnNode& node)
{
    std::cout << indent() << "OnNode: operator "
              << (node.isExtend() ? "Extend" : "On") << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(TrinaryIfNode&)
{
    std::cout << indent() << "TrinaryIfNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(VarDeclNode& node)
{
    std::cout << indent() << "VarDeclNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(VarAssignNode& node)
{
    std::cout << indent() << "VarAssignNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(VarNode& node)
{
    std::cout << indent() << "VarNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(FuncRefNode& node)
{
    std::cout << indent() << "FuncRefNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(JustAnIdentifierNode& node)
{
    std::cout << indent() << "JustAnIdentifierNode: " << node.name() << '\n';
}

void PrintASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << indent() << "FuncArgsDeclNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncDeclNode& node)
{
    std::cout << indent() << "FuncDeclNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(FuncStartNode& node)
{
    std::cout << indent() << "FuncStartNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(FuncAssignNode&)
{
    std::cout << indent() << "FuncAssignNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncArgsNode&)
{
    std::cout << indent() << "FuncArgsNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << indent() << "ReturnStatementNode\n";
    ++indent_;
}

void PrintASTVisitor::visit(FuncCallNode& node)
{
    std::cout << indent() << "FuncCallNode: " << node.name() << '\n';
    ++indent_;
}

void PrintASTVisitor::visit(FuncCallStatementNode&)
{
    std::cout << indent() << "FuncCallStatementNode\n";
    ++indent_;
}


void PrintASTVisitor::visit(LoopNode& node)
{
    std::cout << indent() << "LoopNode: For " << node.loopVariable() << " In " << node.loopSet() << "\n";
    ++indent_;
}


void PrintASTVisitor::visit(ArrayNode& node)
{
    std::cout << indent() << "ArrayNode: array size = " << node.expressionList()->arguments().size() << "\n";
    ++indent_;
}


void PrintASTVisitor::visit(RandomAccessNode& node)
{
    std::cout << indent() << "RandomAccessNode: index = " << node.index() << "\n";
    ++indent_;
}




void PrintASTVisitor::midVisit(SequenceNode&)
{
}

void PrintASTVisitor::postVisit(SequenceNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(QuantityNode&)
{
    --indent_;
}

void PrintASTVisitor::midVisit(BinaryOpNode&)
{
}

void PrintASTVisitor::postVisit(BinaryOpNode&)
{
    --indent_;
}

void PrintASTVisitor::midVisit(MultiplyAddNode& node)
{
}

void PrintASTVisitor::postVisit(MultiplyAddNode& node)
{
}

void PrintASTVisitor::midVisit(MultiplyDivideNode& node)
{
}

void PrintASTVisitor::postVisit(MultiplyDivideNode& node)
{
}

void PrintASTVisitor::midVisit(ComparisonOpNode&)
{
}

void PrintASTVisitor::postVisit(ComparisonOpNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(NormNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(UnaryNegationNode&)
{
    --indent_;
}

void PrintASTVisitor::midVisit(OnNode&)
{
}

void PrintASTVisitor::postVisit(OnNode&)
{
    --indent_;
}

void PrintASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
}

void PrintASTVisitor::colonVisit(TrinaryIfNode&)
{
}

void PrintASTVisitor::postVisit(TrinaryIfNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(VarDeclNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(VarAssignNode&)
{
    --indent_;
}

void PrintASTVisitor::midVisit(FuncArgsDeclNode&)
{
}

void PrintASTVisitor::postVisit(FuncArgsDeclNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncDeclNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncStartNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncAssignNode&)
{
    --indent_;
}

void PrintASTVisitor::midVisit(FuncArgsNode&)
{
}

void PrintASTVisitor::postVisit(FuncArgsNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(ReturnStatementNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncCallNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(FuncCallStatementNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(LoopNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(ArrayNode&)
{
    --indent_;
}

void PrintASTVisitor::postVisit(RandomAccessNode&)
{
    --indent_;
}

void PrintASTVisitor::visit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintASTVisitor::midVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintASTVisitor::postVisit(StencilAssignmentNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintASTVisitor::visit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintASTVisitor::postVisit(StencilNode& node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}


std::string PrintASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}
