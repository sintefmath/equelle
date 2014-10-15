
// This program was created by the Equelle compiler from SINTEF.

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <array>

#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/CartesianGrid.hpp"//Should be renamed EquelleCartesianRuntimeCPU

void ensureRequirements(const equelle::EquelleRuntimeCPU& er);
void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er, equelle::CartesianEquelleRuntime& er_cart);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::CartesianEquelleRuntime er_cart(param);
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er, er_cart);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er,
                          equelle::CartesianEquelleRuntime& er_cart) {
    using namespace equelle;
    ensureRequirements(er);
    (void)er_cart; // To suppress compile warnings if not used below.

    // ============= Generated code starts here ================

    const auto a = CollOfScalar(er.centroid(er.allCells()).col(0));
    const auto b = CollOfScalar(er.centroid(er.allCells()).col(1));
    er.output("hmmm", er.trinaryIf((a > er.operatorExtend(double(0), er.allCells())), (a + b), er.operatorExtend(double(0), er.allCells())));
    const auto a1 = er.operatorOn((a + b), er.allCells(), er.interiorCells());
    const auto b1 = er.operatorOn(b, er.allCells(), er.interiorCells());
    const auto c = er.operatorExtend((a1 + b1), er.interiorCells(), er.allCells());
    const auto array = makeArray((a1 + b1), (a1 - b1), a1);
    const auto qww = "This is a string with \"quoted escapes\" and others \n\n\n such as newlines";
    er.output(qww, double(2));

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
    er.ensureGridDimensionMin(2);
}
