
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

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCPU er(param);

    // ============= Generated code starts here ================

    const Scalar k = er.userSpecifiedScalarWithDefault("k", double(0.3));
    const Scalar dt = er.userSpecifiedScalarWithDefault("dt", double(0.5));
    const CollOfScalar u0 = er.userSpecifiedCollectionOfScalar("u0", er.allCells());
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    auto computeResidual = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux(u);
        const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(ifluxes)));
        return residual;
    };
    const CollOfScalar explicitu = (u0 - computeResidual(u0));
    const CollOfScalar u = er.newtonSolve(computeResidual, u0);
    er.output("explicitu", explicitu);
    er.output("u", u);

    // ============= Generated code ends here ================

    return 0;
}
