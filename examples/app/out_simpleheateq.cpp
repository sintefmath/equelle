
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

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);
void ensureRequirements(const equelle::EquelleRuntimeCPU& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

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

    const Scalar k = er.inputScalarWithDefault("k", double(0.3));
    const Scalar dt = er.inputScalarWithDefault("dt", double(0.5));
    const CollOfScalar u0 = er.inputCollectionOfScalar("u0", er.allCells());
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux_i0_ = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    auto computeInteriorFlux_i2_ = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    auto computeResidual_i1_ = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux_i2_(u);
        const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(ifluxes)));
        return residual;
    };
    auto computeResidual_i3_ = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux_i2_(u);
        const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(ifluxes)));
        return residual;
    };
    const CollOfScalar explicitu = (u0 - computeResidual_i1_(u0));
    const CollOfScalar u = er.newtonSolve(computeResidual_i3_, u0);
    er.output("explicitu", explicitu);
    er.output("u", u);

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
