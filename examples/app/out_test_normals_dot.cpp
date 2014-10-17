
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

    const CollOfVector n = er.normal(er.allFaces());
    const CollOfScalar n2 = er.dot(n, n);
    const CollOfScalar n0 = CollOfScalar(n.col(0));
    const std::tuple<CollOfScalar, CollOfScalar> narray = makeArray(n0, (n0 + n2));
    er.output("squared normals", n2);
    er.output("first component", n0);
    er.output("their sum", std::get<1>(narray));
    auto getsecond = [&](const std::tuple<CollOfScalar, CollOfScalar>& a) -> CollOfScalar {
        return std::get<1>(a);
    };
    er.output("second element of array", getsecond(narray));
    er.output("second element of a different, inline array", getsecond(makeArray(n0, ((double(2) * n0) + n2))));
    er.output("second element of the same, inline array, direct access", std::get<1>(makeArray(n0, ((double(2) * n0) + n2))));
    const CollOfVector q1 = (n * double(3));
    const CollOfVector q2 = (double(3) * n);
    const CollOfVector q3 = (n2 * n);
    const CollOfVector q4 = (n * n2);
    er.output("should be zero", er.norm((((q1 - q2) + q3) - q4)));
    er.output("should be all 1.5", er.sqrt((double(2.25) * n2)));

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(1);
}
