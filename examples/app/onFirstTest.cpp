
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

    const CollOfScalar a = er.operatorExtend(double(1.4), er.allCells());
    const CollOfCell first = er.firstCell(er.boundaryCells());
    const CollOfCell second = er.secondCell(er.boundaryCells());
    const CollOfScalar b = er.trinaryIf(er.isEmpty(second), er.operatorOn(a, er.allCells(), first), er.operatorOn(a, er.allCells(), second));
    er.output("b", b);

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCPU& er)
{
    (void)er;
}
