
// This program was created by the Equelle compiler from SINTEF.

#include "equelle/EquelleRuntimeCPU.hpp"

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);
void ensureRequirements(const equelle::EquelleRuntimeCPU& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {
    using namespace equelle;
    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar a = CollOfScalar(er.centroid(er.allCells()).col(0));
    const CollOfScalar b = CollOfScalar(er.centroid(er.allCells()).col(1));
    er.output("hmmm", er.trinaryIf((a > er.operatorExtend(double(0), er.allCells())), (a + b), er.operatorExtend(double(0), er.allCells())));
    const CollOfScalar a1 = er.operatorOn((a + b), er.allCells(), er.interiorCells());
    const CollOfScalar b1 = er.operatorOn(b, er.allCells(), er.interiorCells());
    const CollOfScalar c = er.operatorExtend((a1 + b1), er.interiorCells(), er.allCells());
    const std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> array = makeArray((a1 + b1), (a1 - b1), a1);
    const String qww = "This is a string with \"quoted escapes\" and others \n\n\n such as newlines";
    er.output(qww, double(2));

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
    er.ensureGridDimensionMin(2);
}
