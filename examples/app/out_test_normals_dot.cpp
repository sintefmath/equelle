
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

    const CollOfVector n = er.normal(er.allFaces());
    const CollOfScalar n2 = er.dot(n, n);
    const CollOfScalar n0 = CollOfScalar(n.col(0));
    er.output("squared normals", n2);
    er.output("first component", n0);

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
}
