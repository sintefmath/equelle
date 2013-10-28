
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

    const CollOfScalar perm = er.userSpecifiedCollectionOfScalar("perm", er.allCells());
    const CollOfScalar poro = er.userSpecifiedCollectionOfScalar("poro", er.allCells());
    const CollOfScalar s0 = er.userSpecifiedCollectionOfScalar("s0", er.allCells());
    const CollOfScalar s = s0;
    const CollOfScalar p = er.operatorOn(double(0), er.allCells());
    const CollOfScalar vol = er.norm(er.allCells());
    auto computeTransmissibilities = [&](const CollOfScalar& permeability) -> CollOfScalar {
        const CollOfFace interior_faces = er.interiorFaces();
        const CollOfCell first = er.firstCell(interior_faces);
        const CollOfCell second = er.secondCell(interior_faces);
        const CollOfVector cdiff1 = (er.centroid(first) - er.centroid(interior_faces));
        const CollOfVector cdiff2 = (er.centroid(second) - er.centroid(interior_faces));
        const CollOfScalar p1 = er.operatorOn(permeability, er.allCells(), first);
        const CollOfScalar p2 = er.operatorOn(permeability, er.allCells(), second);
        const CollOfScalar a = er.norm(interior_faces);
        const CollOfScalar halftrans1 = ((a * p1) * (er.dot(er.normal(interior_faces), cdiff1) / er.dot(cdiff1, cdiff1)));
        const CollOfScalar halftrans2 = ((a * p2) * (er.dot(er.normal(interior_faces), cdiff2) / er.dot(cdiff2, cdiff2)));
        const CollOfScalar trans = (double(1) / ((double(1) / halftrans1) + (double(1) / halftrans2)));
        return trans;
    };
    auto upwind = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux > double(0)), x1, x2);
    };
    const CollOfScalar trans = computeTransmissibilities(perm);
    auto computeFluxes = [&](const CollOfScalar& pp) -> CollOfScalar {
        return (-trans * er.gradient(p));
    };
    auto computeResidual = [&](const CollOfScalar& u, const CollOfScalar& u0, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar fluxes = computeFluxes(u);
        return ((u - u0) + ((dt / vol) * er.divergence(fluxes)));
    };
    const SeqOfScalar timesteps = er.userSpecifiedSequenceOfScalar("timesteps");
    const CollOfScalar u_initial = er.userSpecifiedCollectionOfScalar("u_initial", er.allCells());
    CollOfScalar u0;
    u0 = u_initial;
    for (const Scalar& dt : timesteps) {
        auto computeResidualLocal = [&](const CollOfScalar& u) -> CollOfScalar {
            return computeResidual(u, u0, dt);
        };
        const CollOfScalar u_guess = u0;
        const CollOfScalar u = er.newtonSolve(computeResidualLocal, u_guess);
        er.output("u", u);
        u0 = u;
    }

    // ============= Generated code ends here ================

    return 0;
}
