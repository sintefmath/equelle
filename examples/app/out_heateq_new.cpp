
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

    const Scalar k = (er.inputScalarWithDefault("k", double(2.85)) * double(1));
    const Scalar cv = (er.inputScalarWithDefault("cv", double(2170000)) * double(1));
    const CollOfFace ifaces = er.interiorFaces();
    const CollOfCell first = er.firstCell(ifaces);
    const CollOfCell second = er.secondCell(ifaces);
    const CollOfScalar itrans = (k * (er.norm(ifaces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    const CollOfFace dir_boundary = er.inputDomainSubsetOf("dir_boundary", er.boundaryFaces());
    const CollOfScalar dir_val = (er.inputCollectionOfScalar("dir_val", dir_boundary) * double(1));
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const CollOfCell dir_cells = er.operatorOn(bf_cells, er.boundaryFaces(), dir_boundary);
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dir_boundary);
    const CollOfScalar dir_trans = er.operatorOn(btrans, er.boundaryFaces(), dir_boundary);
    auto computeBoundaryFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), dir_cells);
        const CollOfScalar dir_fluxes = ((dir_trans * dir_sign) * (u_dirbdycells - dir_val));
        return er.operatorExtend(dir_fluxes, dir_boundary, er.boundaryFaces());
    };
    const CollOfScalar vol = er.norm(er.allCells());
    auto computeResidual = [&](const CollOfScalar& u, const CollOfScalar& u0, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux(u);
        const CollOfScalar bfluxes = computeBoundaryFlux(u);
        const CollOfScalar fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
        const CollOfScalar residual = ((u - u0) + ((dt / (cv * vol)) * er.divergence(fluxes)));
        return residual;
    };
    const CollOfScalar u_initial = (er.inputCollectionOfScalar("u_initial", er.allCells()) * double(1));
    const SeqOfScalar timesteps = (er.inputSequenceOfScalar("timesteps") * double(1));
    auto u0 = u_initial;
    for (const Scalar& dt : timesteps) {
        auto computeResidualLocal = [&](const CollOfScalar& u) -> CollOfScalar {
            return computeResidual(u, u0, dt);
        };
        const CollOfScalar u_guess = u0;
        const CollOfScalar u = er.newtonSolve(computeResidualLocal, u_guess);
        er.output("u", u);
        er.output("maximum of u", er.maxReduce(u));
        u0 = u;
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
