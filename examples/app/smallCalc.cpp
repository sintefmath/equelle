
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

void ensureRequirements(const EquelleRuntimeCUDA& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCUDA er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());
    const CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());
    const CollOfScalar d = er.inputCollectionOfScalar("d", er.allCells());
    const CollOfScalar c = (a - b);
    const CollOfScalar e = (d - c);
    er.output("e", e);
    const CollOfScalar f = (e / d);
    er.output("f", f);
    const CollOfScalar g = (f + b);
    er.output("g", g);
    const CollOfScalar h = (g * d);
    er.output("h", h);
    const CollOfScalar i = (h + double(100));

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}
