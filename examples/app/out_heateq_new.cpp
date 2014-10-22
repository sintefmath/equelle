
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

    const Scalar k__ = er.inputScalarWithDefault("k", double(0.3));
    const Scalar k = (k__ * double(1));
    const Scalar cv__ = er.inputScalarWithDefault("cv", double(1));
    const Scalar cv = (cv__ * double(1));
    const CollOfFace ifaces = er.interiorFaces();
    const CollOfCell first = er.firstCell(ifaces);
    const CollOfCell second = er.secondCell(ifaces);
    const CollOfScalar itrans = (k * (er.norm(ifaces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux_i0_ = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    const CollOfFace dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const CollOfScalar dirichlet_val__ = er.inputCollectionOfScalar("dirichlet_val", dirichlet_boundary);
    const CollOfScalar dirichlet_val = (dirichlet_val__ * double(1));
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const CollOfCell dir_cells = er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary);
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);
    const CollOfScalar dir_trans = er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary);
    auto computeBoundaryFlux_i1_ = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), dir_cells);
        const CollOfScalar dir_fluxes = ((dir_trans * dir_sign) * (u_dirbdycells - dirichlet_val));
        return er.operatorExtend(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    const CollOfScalar vol = er.norm(er.allCells());
    auto computeResidual_i2_ = [&](const CollOfScalar& u, const CollOfScalar& u0, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux_i0_(u);
        const CollOfScalar bfluxes = computeBoundaryFlux_i1_(u);
        const CollOfScalar fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
        const CollOfScalar residual = ((u - u0) + ((dt / (cv * vol)) * er.divergence(fluxes)));
        return residual;
    };
    const CollOfScalar u_initial__ = er.inputCollectionOfScalar("u_initial", er.allCells());
    const CollOfScalar u_initial = (u_initial__ * double(1));
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    auto u0 = u_initial;
    for (const Scalar& dt__ : timesteps) {
        const Scalar dt = (dt__ * double(1));
        auto computeResidualLocal_i3_ = [&](const CollOfScalar& u) -> CollOfScalar {
            return computeResidual_i2_(u, u0, dt);
        };
        const CollOfScalar u_guess = u0;
        const CollOfScalar u = er.newtonSolve(computeResidualLocal_i3_, u_guess);
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
