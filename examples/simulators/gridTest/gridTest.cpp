
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

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void ensureRequirements(const EquelleRuntimeCUDA& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCUDA er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar a = er.operatorExtend(double(1), er.allCells());
    const CollOfScalar b = er.operatorExtend(double(2), er.interiorCells());
    const CollOfScalar c = er.operatorExtend(double(3), er.boundaryCells());
    const CollOfScalar d = ((a + er.operatorExtend(b, er.interiorCells(), er.allCells())) + er.operatorExtend(c, er.boundaryCells(), er.allCells()));
    er.output("a", a);
    er.output("b", b);
    er.output("c", c);
    er.output("d", d);

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}
