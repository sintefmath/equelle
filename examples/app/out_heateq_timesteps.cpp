
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
    const auto ifaces = er.interiorFaces();
    const auto first = er.firstCell(ifaces);
    const auto second = er.secondCell(ifaces);
    const auto itrans = (k * (er.norm(ifaces) / er.norm((er.centroid(first) - er.centroid(second)))));
    auto computeInteriorFlux = [&](const auto& u) {
        return (-itrans * er.gradient(u));
    };
    const auto dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const auto dirichlet_val = er.inputCollectionOfScalar("dirichlet_val", dirichlet_boundary);
    const auto bf = er.boundaryFaces();
    const auto bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const auto bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const auto btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const auto dir_cells = er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary);
    const auto dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);
    const auto dir_trans = er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary);
    auto computeBoundaryFlux = [&](const auto& u) {
        const auto u_dirbdycells = er.operatorOn(u, er.allCells(), dir_cells);
        const auto dir_fluxes = ((dir_trans * dir_sign) * (u_dirbdycells - dirichlet_val));
        return er.operatorExtend(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    const auto vol = er.norm(er.allCells());
    auto computeResidual = [&](const auto& u, const auto& u0, const auto& dt) {
        const auto ifluxes = computeInteriorFlux(u);
        const auto bfluxes = computeBoundaryFlux(u);
        const auto fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
        const auto residual = ((u - u0) + ((dt / vol) * er.divergence(fluxes)));
        return residual;
    };
    const auto u_initial = er.inputCollectionOfScalar("u_initial", er.allCells());
    const auto timesteps = er.inputSequenceOfScalar("timesteps");
    auto u0 = u_initial;
    for (const Scalar& dt : timesteps) {
        auto computeResidualLocal = [&](const auto& u) {
            return computeResidual(u, u0, dt);
        };
        const auto u_guess = u0;
        const auto u = er.newtonSolve(computeResidualLocal, u_guess);
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
