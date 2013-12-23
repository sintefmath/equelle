
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

#include "EquelleRuntimeCPU.hpp"

void ensureRequirements(const EquelleRuntimeCPU& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCPU er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar a = CollOfScalar(er.centroid(er.allCells()).col(0));
    const CollOfScalar b = CollOfScalar(er.centroid(er.allCells()).col(1));
    er.output("hmmm", er.trinaryIf((a > er.operatorExtend(double(0), er.allCells())), (a + b), er.operatorExtend(double(0), er.allCells())));
    const CollOfScalar a1 = er.operatorOn((a + b), er.allCells(), er.interiorCells());
    const CollOfScalar b1 = er.operatorOn(b, er.allCells(), er.interiorCells());
    const CollOfScalar c = er.operatorExtend((a1 + b1), er.interiorCells(), er.allCells());
    const std::array<CollOfScalar, 3> array = makeArray((a1 + b1), (a1 - b1), a1);
    const String qww = "This is a string with \"quoted escapes\" and others \n\n\n such as newlines";
    er.output(qww, double(2));

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
    er.ensureGridDimensionMin(2);
}
