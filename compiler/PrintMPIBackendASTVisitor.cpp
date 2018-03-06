#include "PrintMPIBackendASTVisitor.hpp"

namespace
{
    const char* impl_cppStartString()
    {
        return
"\n"
"// This program was created by the Equelle compiler from SINTEF.\n"
"\n"
"#include <opm/core/utility/parameters/ParameterGroup.hpp>\n"
"#include <opm/core/linalg/LinearSolverFactory.hpp>\n"
"#include <opm/common/ErrorMacros.hpp>\n"
"#include <opm/autodiff/AutoDiffBlock.hpp>\n"
"#include <opm/autodiff/AutoDiffHelpers.hpp>\n"
"#include <opm/core/grid.h>\n"
"#include <opm/core/grid/GridManager.hpp>\n"
"#include <algorithm>\n"
"#include <iterator>\n"
"#include <iostream>\n"
"#include <cmath>\n"
"#include <array>\n"
"\n"
"#include \"equelle/RuntimeMPI.hpp\"\n"
"\n"
"//void ensureRequirements(const equelle::RuntimeMPI& er);\n"
"void equelleGeneratedCode(equelle::RuntimeMPI& er);\n"
"\n"
 "#ifndef EQUELLE_NO_MAIN\n"
"int main(int argc, char** argv)\n"
"{\n"
"    // Initialize MPI\n"
"    equelle::MPIInitializer mpiInitializer(argc, argv);"
"    // Get user parameters.\n"
"    Opm::parameter::ParameterGroup param(argc, argv, false);\n"
"\n"
"    // Create the Equelle runtime.\n"
"    equelle::RuntimeMPI er(param);\n"
"    er.decompose();\n"
"    MPI_SAFE_CALL( MPI_Barrier( MPI_COMM_WORLD ) );\n"
"    auto startTime = MPI_Wtime();\n"
"    equelleGeneratedCode(er);\n"
"    MPI_SAFE_CALL( MPI_Barrier( MPI_COMM_WORLD ) );\n"
"    auto endTime = MPI_Wtime();\n"
"    er.logstream << \"Running time for generated code: \" << endTime-startTime << std::endl;\n"
"    return 0;\n"
"}\n"
"#endif // EQUELLE_NO_MAIN\n"
"\n"
"void equelleGeneratedCode(equelle::RuntimeMPI& er) {\n"
"    using namespace equelle;\n"
"    //ensureRequirements(er);\n"
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


PrintMPIBackendASTVisitor::PrintMPIBackendASTVisitor()
{
}

PrintMPIBackendASTVisitor::~PrintMPIBackendASTVisitor()
{

}

const char *PrintMPIBackendASTVisitor::cppStartString() const
{
    return ::impl_cppStartString();
}

const char *PrintMPIBackendASTVisitor::cppEndString() const
{
    return ::impl_cppEndString();
}

const char *PrintMPIBackendASTVisitor::classNameString() const
{
    return "RuntimeMPI";
}

const char *PrintMPIBackendASTVisitor::namespaceNameString() const
{
    return "equelle";
}
