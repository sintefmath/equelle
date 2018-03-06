/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "PrintCPUBackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <cctype>
#include <sstream>
#include <stdexcept>


namespace
{
    const char* impl_cppStartString();
    const char* impl_cppCartesianStartString();
    const char* impl_cppEndString();
}

PrintCPUBackendASTVisitor::PrintCPUBackendASTVisitor()
    : suppression_level_(0),
      indent_(1),
      sequence_depth_(0),
      instantiating_(false),
      next_funcstart_inst_(-1),
      use_cartesian_(false)
{
}

PrintCPUBackendASTVisitor::PrintCPUBackendASTVisitor(const bool use_cartesian)
    : suppression_level_(0),
      indent_(1),
      sequence_depth_(0),
      instantiating_(false),
      next_funcstart_inst_(-1),
      use_cartesian_(use_cartesian)
{
}

PrintCPUBackendASTVisitor::~PrintCPUBackendASTVisitor()
{
}

void PrintCPUBackendASTVisitor::visit(SequenceNode&)
{
    if (sequence_depth_ == 0) {
        // This is the root node of the program.
        std::cout << cppStartString();
        endl();
    }
    ++sequence_depth_;
}

void PrintCPUBackendASTVisitor::midVisit(SequenceNode&)
{
}

void PrintCPUBackendASTVisitor::postVisit(SequenceNode&)
{
    --sequence_depth_;
    if (sequence_depth_ == 0) {
        // We are back at the root node. Finish main() function.
        std::cout << cppEndString();
        // Emit ensureRequirements() function.
        std::cout <<
            "\n"
            "void ensureRequirements(const " << namespaceNameString() <<
	    "::" << classNameString() << "& er)\n"
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

void PrintCPUBackendASTVisitor::visit(NumberNode& node)
{
    if (isSuppressed()) {
        return;
    }
    std::cout.precision(16);
    std::cout << "double(" << node.number() << ")";
}

void PrintCPUBackendASTVisitor::visit(QuantityNode& node)
{
    if (isSuppressed()) {
        return;
    }
    const double cf = node.conversionFactorSI();
    if (cf != 1.0) {
        std::cout.precision(16);
        std::cout << "(" << cf << "*";
    }
}

void PrintCPUBackendASTVisitor::postVisit(QuantityNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (node.conversionFactorSI() != 1.0) {
        std::cout << ")";
    }
}

void PrintCPUBackendASTVisitor::visit(StringNode& node)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << node.content();
}

void PrintCPUBackendASTVisitor::visit(TypeNode&)
{
    // std::cout << SymbolTable::equelleString(node.type());
}

void PrintCPUBackendASTVisitor::visit(FuncTypeNode&)
{
    // std::cout << node.funcType().equelleString();
}

void PrintCPUBackendASTVisitor::visit(BinaryOpNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << '(';
}

void PrintCPUBackendASTVisitor::midVisit(BinaryOpNode& node)
{
    if (isSuppressed()) {
        return;
    }
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

void PrintCPUBackendASTVisitor::postVisit(BinaryOpNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(MultiplyAddNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << "er.multiplyAdd(";
}

void PrintCPUBackendASTVisitor::midVisit(MultiplyAddNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ", ";
}

void PrintCPUBackendASTVisitor::postVisit(MultiplyAddNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}


void PrintCPUBackendASTVisitor::visit(ComparisonOpNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << '(';
}

void PrintCPUBackendASTVisitor::midVisit(ComparisonOpNode& node)
{
    if (isSuppressed()) {
        return;
    }
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

void PrintCPUBackendASTVisitor::postVisit(ComparisonOpNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(NormNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << "er.norm(";
}

void PrintCPUBackendASTVisitor::postVisit(NormNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(UnaryNegationNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << '-';
}

void PrintCPUBackendASTVisitor::postVisit(UnaryNegationNode&)
{
}

void PrintCPUBackendASTVisitor::visit(OnNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (node.isExtend()) {
        std::cout << "er.operatorExtend(";
    } else {
        std::cout << "er.operatorOn(";
    }
}

void PrintCPUBackendASTVisitor::midVisit(OnNode& node)
{
    if (isSuppressed()) {
        return;
    }
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
        const std::string cppterm = std::isupper(first) ?
            std::string("er.") + char(std::tolower(first)) + esname.substr(1)
            : esname;
        std::cout << cppterm;
        std::cout << ", ";
    }
}

void PrintCPUBackendASTVisitor::postVisit(OnNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(TrinaryIfNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << "er.trinaryIf(";
}

void PrintCPUBackendASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ", ";
}

void PrintCPUBackendASTVisitor::colonVisit(TrinaryIfNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ", ";
}

void PrintCPUBackendASTVisitor::postVisit(TrinaryIfNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(VarDeclNode& node)
{
    suppress();
}

void PrintCPUBackendASTVisitor::postVisit(VarDeclNode&)
{
    unsuppress();
}

void PrintCPUBackendASTVisitor::visit(VarAssignNode& node)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << indent();
    if (node.type() == StencilI || node.type() == StencilJ || node.type() == StencilK) {
        //This goes into the stencil-lambda definition, and is only used during parsing.
        std::cout << "// Note: ";
    }
    if (!SymbolTable::variableType(node.name()).isMutable()) {
#if 0
        std::cout << "const auto ";
#else
        std::cout << "const " << cppTypeString(node.type()) << " ";
#endif
    } else if (defined_mutables_.count(node.name()) == 0) {
        std::cout << "auto ";
        defined_mutables_.insert(node.name());
    }
    std::cout << node.name() << " = ";
}

void PrintCPUBackendASTVisitor::postVisit(VarAssignNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ';';
    endl();
}

void PrintCPUBackendASTVisitor::visit(VarNode& node)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << node.name();
    if (SymbolTable::isFunctionDeclared(node.name())) {
        if (SymbolTable::getFunction(node.name()).isTemplate()) {
            const int num_inst = SymbolTable::getFunction(node.name()).instantiations().size();
            assert(num_inst > 0);
            assert(node.instantiationIndex() >= 0);
            if (num_inst > 1) {
                std::cout << "_i" << node.instantiationIndex() << "_";
            }
        }
    }
}

void PrintCPUBackendASTVisitor::visit(FuncRefNode& node)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << node.name();
}

void PrintCPUBackendASTVisitor::visit(JustAnIdentifierNode&)
{
}

void PrintCPUBackendASTVisitor::visit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::visit()}";
}

void PrintCPUBackendASTVisitor::midVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::midVisit()}";
}

void PrintCPUBackendASTVisitor::postVisit(FuncArgsDeclNode&)
{
    std::cout << "{FuncArgsDeclNode::postVisit()}";
}

void PrintCPUBackendASTVisitor::visit(FuncDeclNode&)
{
    // std::cout << node.name() << " : ";
}

void PrintCPUBackendASTVisitor::postVisit(FuncDeclNode&)
{
    // endl();
}

void PrintCPUBackendASTVisitor::visit(FuncStartNode& node)
{
    if (isSuppressed()) {
        suppress();
        return;
    }
    const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    const size_t n = ft.arguments().size();
    std::cout << indent() << "auto " << node.name();
    const int num_inst = SymbolTable::getFunction(node.name()).instantiations().size();
    if (next_funcstart_inst_ != -1 && num_inst > 1) {
        std::cout << "_i" << next_funcstart_inst_ << "_";
    }
    std::cout << " = [&](";
    for (int i = 0; i < n; ++i) {
#if 0
        std::cout << "const auto& " << ft.arguments()[i].name();
#else
        std::cout << "const "
                  << cppTypeString(ft.arguments()[i].type())
                  << "& " << ft.arguments()[i].name();
#endif
        if (i < n - 1) {
            std::cout << ", ";
        }
    }
    suppress();
    ++indent_;
}

void PrintCPUBackendASTVisitor::postVisit(FuncStartNode& node)
{
    next_funcstart_inst_ = -1;
    unsuppress();
    if (isSuppressed()) {
        return;
    }
#if 0
    std::cout << ") {";
#else
    const FunctionType& ft = SymbolTable::getFunction(node.name()).functionType();
    std::cout << ") -> " << cppTypeString(ft.returnType()) << " {";
#endif
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncAssignNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (instantiating_) {
        return;
    }
    Function& f = SymbolTable::getMutableFunction(node.name());
    const auto insta = f.instantiations(); // a copy: important for below
    const int num_inst = insta.size();
    if (num_inst > 0) {
        assert(f.isTemplate());
        instantiating_ = true;
        // All but the last instantiation are done through
        // accept() below, the last one follows the regular
        // visitor flow.
        for (int inst = 0; inst < num_inst - 1; ++inst) {
            f = SymbolTable::getFunctionInstantiation(insta[inst]);
            f.setInstantiations(insta); // here is where we need insta to be a true copy
            next_funcstart_inst_ = insta[inst];
            SymbolTable::setCurrentFunction(node.name());
            node.accept(*this);
        }
        instantiating_ = false;
        f = SymbolTable::getFunctionInstantiation(insta.back());
        next_funcstart_inst_ = insta.back();
    } else {
        skipping_function_ = node.name();
        suppress();
    }
    SymbolTable::setCurrentFunction(node.name());
}

void PrintCPUBackendASTVisitor::postVisit(FuncAssignNode& node)
{
    if (skipping_function_ == node.name()) {
        skipping_function_ = "";
        unsuppress();
        SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
        return;
    }
    if (isSuppressed()) {
        return;
    }
    --indent_;
    std::cout << indent() << "};";
    endl();
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
}

void PrintCPUBackendASTVisitor::visit(FuncArgsNode&)
{
}

void PrintCPUBackendASTVisitor::midVisit(FuncArgsNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ", ";
}

void PrintCPUBackendASTVisitor::postVisit(FuncArgsNode&)
{
}

void PrintCPUBackendASTVisitor::visit(ReturnStatementNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << indent() << "return ";
}

void PrintCPUBackendASTVisitor::postVisit(ReturnStatementNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ';';
    endl();
}

void PrintCPUBackendASTVisitor::visit(FuncCallNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (SymbolTable::isFunctionDeclared(node.name())) {
        const std::string fname = node.name();
        const char first = fname[0];
        std::string cppname;
        if (std::isupper(first)) {
#if 0
            bool is_stencil = false;
            is_stencil = is_stencil | node.type().isStencil();
            const std::vector<EquelleType>& types = node.args()->argumentTypes();
            for (int i=0; i<types.size(); ++i) {
                is_stencil = is_stencil | types[i].isStencil();
            }
            if (is_stencil) {
                cppname += std::string("er_cart.");
            }
            else {
                cppname += std::string("er.");
            }
#endif
            cppname += std::string("er.");
            cppname += char(std::tolower(first)) + fname.substr(1);
        } else {
            cppname += fname;
        }
        const int num_inst = SymbolTable::getFunction(node.name()).instantiations().size();
        if (node.instantiationIndex() >= 0 && num_inst > 1) {
            cppname += "_i";
            cppname += std::to_string(node.instantiationIndex());
            cppname += "_";
        }
        std::cout << cppname << '(';
    }
    else if (SymbolTable::isVariableDeclared(node.name()) && node.type().isStencil()) {
        std::cout << "grid.cellAt( " << node.name() << ", ";
    }
    else {
        //Error here?
    }
}

void PrintCPUBackendASTVisitor::postVisit(FuncCallNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ')';
}

void PrintCPUBackendASTVisitor::visit(FuncCallStatementNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << indent();
}

void PrintCPUBackendASTVisitor::postVisit(FuncCallStatementNode&)
{
    if (isSuppressed()) {
        return;
    }
    std::cout << ';';
    endl();
}

void PrintCPUBackendASTVisitor::visit(LoopNode& node)
{
    if (isSuppressed()) {
        return;
    }
    SymbolTable::setCurrentFunction(node.loopName());
    BasicType loopvartype = SymbolTable::variableType(node.loopSet()).basicType();
    std::cout << indent() << "for (const " << cppTypeString(loopvartype) << "& "
              << node.loopVariable() << " : " << node.loopSet() << ") {";
    ++indent_;
    endl();
}

void PrintCPUBackendASTVisitor::postVisit(LoopNode&)
{
    if (isSuppressed()) {
        return;
    }
    --indent_;
    std::cout << indent() << "}";
    endl();
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
}

void PrintCPUBackendASTVisitor::visit(ArrayNode&)
{
    if (isSuppressed()) {
        return;
    }
    // std::cout << cppTypeString(node.type()) << "({{";
    std::cout << "makeArray(";
}

void PrintCPUBackendASTVisitor::postVisit(ArrayNode&)
{
    if (isSuppressed()) {
        return;
    }
    // std::cout << "}})";
    std::cout << ")";
}

void PrintCPUBackendASTVisitor::visit(RandomAccessNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (node.arrayAccess()) {
        // This is Array access.
        std::cout << "std::get<" << node.index() << ">(";
    } else {
        // This is Vector access.
        std::cout << "CollOfScalar(";
    }
}

void PrintCPUBackendASTVisitor::postVisit(RandomAccessNode& node)
{
    if (isSuppressed()) {
        return;
    }
    if (node.arrayAccess()) {
        // This is Array access.
        std::cout << ")";
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

const char* PrintCPUBackendASTVisitor::cppStartString() const
{
    if ( use_cartesian_ ) {
	return ::impl_cppCartesianStartString();
    }
    return ::impl_cppStartString();
}

const char* PrintCPUBackendASTVisitor::cppEndString() const
{
    return ::impl_cppEndString();
}

const char* PrintCPUBackendASTVisitor::classNameString() const
{
    return "EquelleRuntimeCPU";
}

const char* PrintCPUBackendASTVisitor::namespaceNameString() const
{
    return "equelle";
}

void PrintCPUBackendASTVisitor::endl() const
{
    std::cout << '\n';
}

std::string PrintCPUBackendASTVisitor::indent() const
{
    return std::string(indent_*4, ' ');
}

void PrintCPUBackendASTVisitor::suppress()
{
    ++suppression_level_;
}

void PrintCPUBackendASTVisitor::unsuppress()
{
    --suppression_level_;
    assert(suppression_level_ >= 0);
}

bool PrintCPUBackendASTVisitor::isSuppressed() const
{
    return suppression_level_ > 0;
}

std::string PrintCPUBackendASTVisitor::cppTypeString(const EquelleType& et) const
{
    std::string cppstring;
    if (et.isArray()) {
        cppstring += "std::tuple<";
        EquelleType basic_et = et;
        basic_et.setArraySize(NotAnArray);
        std::string basiccppstring = cppTypeString(basic_et);
        for (int elem = 0; elem < et.arraySize(); ++elem) {
            cppstring += basiccppstring;
            if (elem < et.arraySize() - 1) {
                cppstring += ", ";
            }
        }
        cppstring += ">";
        return cppstring;
    }
    if (et.isStencil()) {
        cppstring += "Stencil";
    }
    if (et.isCollection()) {
        cppstring += "CollOf";
    } else if (et.isSequence()) {
        cppstring += "SeqOf";
    }
    cppstring += basicTypeString(et.basicType());
    return cppstring;
}

void PrintCPUBackendASTVisitor::addRequirementString(const std::string& req)
{
    requirement_strings_.insert(req);
}


void PrintCPUBackendASTVisitor::visit(StencilAssignmentNode &node)
{
    std::cout << indent() << "{ //Start of stencil-lambda" << std::endl;
    indent_++;
    //FIXME: Make dimension independent
    std::cout << indent() << "auto cell_stencil = [&]( int i, int j ) {" << std::endl;
    indent_++;
    std::cout << indent();
}

void PrintCPUBackendASTVisitor::midVisit(StencilAssignmentNode &node)
{
    std::cout << " = " << std::endl;
    indent_++;
    std::cout << indent();
    indent_--;
}

void PrintCPUBackendASTVisitor::postVisit(StencilAssignmentNode &node)
{
    std::string gridMapping = SymbolTable::entitySetName(node.type().gridMapping());
    gridMapping[0] = tolower(gridMapping[0]);
    indent_--;
    std::cout << ";" << std::endl;
    std::cout << indent() << "};" << std::endl;
    std::cout << indent() << node.name() << ".grid." << gridMapping << ".execute( cell_stencil );" << std::endl;
    indent_--;
    std::cout << indent() << "} // End of stencil-lambda" << std::endl;
}

void PrintCPUBackendASTVisitor::visit(StencilNode& node)
{
    //FIXME If using half indices, should then use faceAt, not cellAt
    std::cout << node.name() << ".grid.cellAt(" << node.name() << ", ";
}

void PrintCPUBackendASTVisitor::postVisit(StencilNode& node)
{
    std::cout << ")";
}


namespace
{
    const char* impl_cppCartesianStartString() {
 return
"\n"
"// This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"#include \"equelle/EquelleRuntimeCPU.hpp\"\n"
"\n"
"void ensureRequirements(const equelle::EquelleRuntimeCPU& er);\n"
"void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);\n"
"\n"
 "#ifndef EQUELLE_NO_MAIN\n"
"int main(int argc, char** argv)\n"
"{\n"
"    // Get user parameters.\n"
"    Opm::ParameterGroup param(argc, argv, false);\n"
"\n"
"    // Create the Equelle runtime.\n"
"    equelle::EquelleRuntimeCPU er(param);\n"
"    equelleGeneratedCode(er);\n"
"    return 0;\n"
"}\n"
"#endif // EQUELLE_NO_MAIN\n"
"\n"
"void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {\n"
"    using namespace equelle;\n"
"    ensureRequirements(er);\n"
"\n"
"    // ============= Generated code starts here ================\n";
    }
    
    const char* impl_cppStartString()
    {
        return
"\n"
"// This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"#include \"equelle/EquelleRuntimeCPU.hpp\"\n"
"\n"
"void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);\n"
"void ensureRequirements(const equelle::EquelleRuntimeCPU& er);\n"
"\n"
 "#ifndef EQUELLE_NO_MAIN\n"
"int main(int argc, char** argv)\n"
"{\n"
"    // Get user parameters.\n"
"    Opm::ParameterGroup param(argc, argv, false);\n"
"\n"
"    // Create the Equelle runtime.\n"
"    equelle::EquelleRuntimeCPU er(param);\n"
"    equelleGeneratedCode(er);\n"
"    return 0;\n"
"}\n"
"#endif // EQUELLE_NO_MAIN\n"
"\n"
"void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {\n"
"    using namespace equelle;\n"
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
