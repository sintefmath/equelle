
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

    const auto k = er.inputScalarWithDefault("k", double(0.3));
    const auto dt = er.inputScalarWithDefault("dt", double(0.5));
    const auto u0 = er.inputCollectionOfScalar("u0", er.allCells());
    const auto vol = er.norm(er.allCells());
    const auto interior_faces = er.interiorFaces();
    const auto first = er.firstCell(interior_faces);
    const auto second = er.secondCell(interior_faces);
    const auto itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux = [&](const auto& u) {
        return (-itrans * er.gradient(u));
    };
    auto computeResidual = [&](const auto& u) {
        const auto ifluxes = computeInteriorFlux(u);
        const auto residual = ((u - u0) + ((dt / vol) * er.divergence(ifluxes)));
        return residual;
    };
    const auto explicitu = (u0 - computeResidual(u0));
    const auto u = er.newtonSolve(computeResidual, u0);
    er.output("explicitu", explicitu);
    er.output("u", u);

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
