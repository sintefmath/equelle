/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintCUDABackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <cctype>
#include <sstream>
#include <stdexcept>


namespace
{
    const char* impl_cppStartString();
    const char* impl_cppEndString();
}



PrintCUDABackendASTVisitor::PrintCUDABackendASTVisitor()
    : suppressed_(false),
      indent_(1),
      sequence_depth_(0)
{
}

PrintCUDABackendASTVisitor::~PrintCUDABackendASTVisitor()
{
}





void PrintCUDABackendASTVisitor::visit(SequenceNode&)
{
    if (sequence_depth_ == 0) {
        // This is the root node of the program.
        std::cout << cppStartString();
        endl();
    }
    ++sequence_depth_;
}

void PrintCUDABackendASTVisitor::midVisit(SequenceNode&)
{
}

void PrintCUDABackendASTVisitor::postVisit(SequenceNode&)
{
    --sequence_depth_;
    if (sequence_depth_ == 0) {
        // We are back at the root node. Finish main() function.
        std::cout << cppEndString();
        // Emit ensureRequirements() function.
        std::cout <<
            "\n"
            "void ensureRequirements(const EquelleRuntimeCUDA& er)\n"
            "{\n";
        if (requirement_strings_.empty()) {
            std::cout << "    (void)er;\n";
        }
        for (const std::string& req : requirement_strings_) {
            std::cout << "    " << req;
        }
        std::cout << 
            "}\n";
    }
}

void PrintCUDABackendASTVisitor::visit(NumberNode& node)
{
    std::cout.precision(16);
    std::cout << "double(" << node.number() << ")";
}

void PrintCUDABackendASTVisitor::visit(StringNode& node)
{
    std::cout << node.content();
}

void PrintCUDABackendASTVisitor::visit(TypeNode&)
{
    // std::cout << SymbolTable::equelleString(node.type());
}

void PrintCUDABackendASTVisitor::visit(FuncTypeNode&)
{
    // std::cout << node.funcType().equelleString();
}

void PrintCUDABackendASTVisitor::visit(BinaryOpNode&)
{
    std::cout << '(';
}

void PrintCUDABackendASTVisitor::midVisit(BinaryOpNode& node)
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

void PrintCUDABackendASTVisitor::postVisit(BinaryOpNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(ComparisonOpNode&)
{
    std::cout << '(';
}

void PrintCUDABackendASTVisitor::midVisit(ComparisonOpNode& node)
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

void PrintCUDABackendASTVisitor::postVisit(ComparisonOpNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(NormNode&)
{
    std::cout << "er.norm(";
}

void PrintCUDABackendASTVisitor::postVisit(NormNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(UnaryNegationNode&)
{
    std::cout << '-';
}

void PrintCUDABackendASTVisitor::postVisit(UnaryNegationNode&)
{
}

void PrintCUDABackendASTVisitor::visit(OnNode& node)
{
    if (node.isExtend()) {
        std::cout << "er.operatorExtend(";
    } else {
        std::cout << "er.operatorOn(";
    }
}

void PrintCUDABackendASTVisitor::midVisit(OnNode& node)
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
            std::string("er.") + char(std::tolower(first)) + esname.substr(1)
            : esname;
        std::cout << cppterm;
        std::cout << ", ";
    }
}

void PrintCUDABackendASTVisitor::postVisit(OnNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(TrinaryIfNode&)
{
    std::cout << "er.trinaryIf(";
}

void PrintCUDABackendASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
    std::cout << ", ";
}

void PrintCUDABackendASTVisitor::colonVisit(TrinaryIfNode&)
{
    std::cout << ", ";
}

void PrintCUDABackendASTVisitor::postVisit(TrinaryIfNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(VarDeclNode& node)
{
    if (node.type().isMutable()) {
        std::cout << indent() << cppTypeString(node.type()) << " " << node.name() << ';';
        endl();
    }
    // suppress();
}

void PrintCUDABackendASTVisitor::postVisit(VarDeclNode&)
{
    // unsuppress();
}

void PrintCUDABackendASTVisitor::visit(VarAssignNode& node)
{
    std::cout << indent();
    if (!SymbolTable::variableType(node.name()).isMutable()) {
        std::cout << "const " << cppTypeString(node.type()) << " ";
    }
    std::cout << node.name() << " = ";
}

void PrintCUDABackendASTVisitor::postVisit(VarAssignNode&)
{
    std::cout << ';';
    endl();
}

void PrintCUDABackendASTVisitor::visit(VarNode& node)
{
    if (!suppressed_) {
        std::cout << node.name();
    }
}

void PrintCUDABackendASTVisitor::visit(FuncRefNode& node)
{
    std::cout << node.name();
}

void PrintCUDABackendASTVisitor::visit(JustAnIdentifierNode&)
{
}

void PrintCUDABackendASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintCUDABackendASTVisitor::midVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::midVisit()}";
}

void PrintCUDABackendASTVisitor::postVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintCUDABackendASTVisitor::visit(FuncDeclNode&)
{
    // std::cout << node.name() << " : ";
}

void PrintCUDABackendASTVisitor::postVisit(FuncDeclNode&)
{
    // endl();
}

void PrintCUDABackendASTVisitor::visit(FuncStartNode& node)
{
    // std::cout << indent() << "auto " << node.name() << " = [&](";
    const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    const size_t n = ft.arguments().size();
    std::cout << indent() << "std::function<" << cppTypeString(ft.returnType()) << '(';
    for (int i = 0; i < n; ++i) {
        std::cout << "const "
                  << cppTypeString(ft.arguments()[i].type())
                  << "&";
        if (i < n - 1) {
            std::cout << ", ";
        }
    }
    std::cout << ")> " << node.name() << " = [&](";
    for (int i = 0; i < n; ++i) {
        std::cout << "const "
                  << cppTypeString(ft.arguments()[i].type())
                  << "& " << ft.arguments()[i].name();
        if (i < n - 1) {
            std::cout << ", ";
        }
    }
    suppress();
    ++indent_;
}

void PrintCUDABackendASTVisitor::postVisit(FuncStartNode& node)
{
    unsuppress();
    const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    std::cout << ") -> " << cppTypeString(ft.returnType()) << " {";
    endl();
}

void PrintCUDABackendASTVisitor::visit(FuncAssignNode&)
{
}

void PrintCUDABackendASTVisitor::postVisit(FuncAssignNode&)
{
    --indent_;
    std::cout << indent() << "};";
    endl();
}

void PrintCUDABackendASTVisitor::visit(FuncArgsNode&)
{
}

void PrintCUDABackendASTVisitor::midVisit(FuncArgsNode&)
{
    if (!suppressed_) {
        std::cout << ", ";
    }
}

void PrintCUDABackendASTVisitor::postVisit(FuncArgsNode&)
{
}

void PrintCUDABackendASTVisitor::visit(ReturnStatementNode&)
{
    std::cout << indent() << "return ";
}

void PrintCUDABackendASTVisitor::postVisit(ReturnStatementNode&)
{
    std::cout << ';';
    endl();
}

void PrintCUDABackendASTVisitor::visit(FuncCallNode& node)
{
    const std::string fname = node.name();
    const char first = fname[0];
    std::string cppname;
    if (std::isupper(first)) {
        cppname += std::string("er.") + char(std::tolower(first)) + fname.substr(1);
    } else {
        cppname += fname;
    }
    // Special treatment for the NewtonSolveSystem() function, since it is unable to
    // deduce its template parameter <int Num>.
    if (fname == "NewtonSolveSystem") {
        std::ostringstream extra;
        extra << "<" << node.type().arraySize() << ">";
        cppname += extra.str();
    }
    std::cout << cppname << '(';
}

void PrintCUDABackendASTVisitor::postVisit(FuncCallNode&)
{
    std::cout << ')';
}

void PrintCUDABackendASTVisitor::visit(FuncCallStatementNode&)
{
    std::cout << indent();
}

void PrintCUDABackendASTVisitor::postVisit(FuncCallStatementNode&)
{
    std::cout << ';';
    endl();
}

void PrintCUDABackendASTVisitor::visit(LoopNode& node)
{
    BasicType loopvartype = SymbolTable::variableType(node.loopSet()).basicType();
    std::cout << indent() << "for (const " << cppTypeString(loopvartype) << "& "
              << node.loopVariable() << " : " << node.loopSet() << ") {";
    ++indent_;
    endl();
}

void PrintCUDABackendASTVisitor::postVisit(LoopNode&)
{
    --indent_;
    std::cout << indent() << "}";
    endl();
}

void PrintCUDABackendASTVisitor::visit(ArrayNode&)
{
    // std::cout << cppTypeString(node.type()) << "({{";
    std::cout << "makeArray(";
}

void PrintCUDABackendASTVisitor::postVisit(ArrayNode&)
{
    // std::cout << "}})";
    std::cout << ")";
}

void PrintCUDABackendASTVisitor::visit(RandomAccessNode& node)
{
    if (!node.arrayAccess()) {
        // This is Vector access.
        std::cout << "CollOfScalar(";
    }
}

void PrintCUDABackendASTVisitor::postVisit(RandomAccessNode& node)
{
    if (node.arrayAccess()) {
        // This is Array access.
        std::cout << "[" << node.index() << "]";
    } else {
        // This is Vector access.
        // Add a grid dimension requirement.
        std::ostringstream os;
        os << "er.ensureGridDimensionMin(" << node.index() + 1 << ");\n";
        addRequirementString(os.str());
        // Random access op is taking the column of the underlying Eigen array.
        std::cout << ".col(" << node.index() << "))";
    }
}

const char *PrintCUDABackendASTVisitor::cppStartString() const
{
    return ::impl_cppStartString();
}

const char *PrintCUDABackendASTVisitor::cppEndString() const
{
    return ::impl_cppEndString();
}

void PrintCUDABackendASTVisitor::endl() const
{
    std::cout << '\n';
}

std::string PrintCUDABackendASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}

void PrintCUDABackendASTVisitor::suppress()
{
    suppressed_ = true;
}

void PrintCUDABackendASTVisitor::unsuppress()
{
    suppressed_ = false;
}

std::string PrintCUDABackendASTVisitor::cppTypeString(const EquelleType& et) const
{
    std::string cppstring;
    if (et.isArray()) {
        cppstring += "std::array<";
    }
    if (et.isCollection()) {
        cppstring += "CollOf";
    } else if (et.isSequence()) {
        cppstring += "SeqOf";
    }
    cppstring += basicTypeString(et.basicType());
    if (et.isArray()) {
        cppstring += ", " + std::to_string(et.arraySize()) + ">";
    }
    return cppstring;
}

void PrintCUDABackendASTVisitor::visit(StencilAccessNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::midVisit(StencilAccessNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::postVisit(StencilAccessNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::visit(StencilStatementNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::midVisit(StencilStatementNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::postVisit(StencilStatementNode &node)
{
    throw std::runtime_error( std::string(__PRETTY_FUNCTION__) + "is not implemented yet" );
}

void PrintCUDABackendASTVisitor::addRequirementString(const std::string& req)
{
    requirement_strings_.insert(req);
}

namespace
{
    const char* impl_cppStartString()
    {
        return
"\n"
"// This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"#include <opm/core/utility/parameters/ParameterGroup.hpp>\n"
"#include <opm/core/utility/ErrorMacros.hpp>\n"
"#include <opm/core/grid.h>\n"
"#include <opm/core/grid/GridManager.hpp>\n"
"#include <algorithm>\n"
"#include <iterator>\n"
"#include <iostream>\n"
"#include <cmath>\n"
"#include <array>\n"
"\n"
"#include \"EquelleRuntimeCUDA.hpp\"\n"
"\n"
"void ensureRequirements(const EquelleRuntimeCUDA& er);\n"
"void equelleGeneratedCode(equelleCUDA::EquelleRuntimeCUDA& er);\n"
"\n"
"#ifndef EQUELLE_NO_MAIN\n"
"int main(int argc, char** argv)\n"
"{\n"
"    // Get user parameters.\n"
"    Opm::parameter::ParameterGroup param(argc, argv, false);\n"
"\n"
"    // Create the Equelle runtime.\n"
"    equelleCUDA::EquelleRuntimeCUDA er(param);\n"
"    equelleGeneratedCode(er);\n"
"    return 0;\n"
"}\n"
"#endif // EQUELLE_NO_MAIN\n"
"\n"
"void equelleGeneratedCode(equelleCUDA::EquelleRuntimeCUDA& er) {\n"
"    using namespace equelleCUDA;\n"
"    ensureRequirements(er);\n"
"\n"
"    // ============= Generated code starts here ================\n";
    }

    const char* impl_cppEndString()
    {
        return "\n"
"    // ============= Generated code ends here ================\n"
"\n"
"}\n";
    }
}
