
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

    const Scalar k = er.inputScalarWithDefault("k", double(0.3));
    const Scalar dt = er.inputScalarWithDefault("dt", double(0.5));
    const CollOfScalar u0 = er.inputCollectionOfScalar("u0", er.allCells());
    const CollOfFace dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const CollOfScalar dirichlet_val = er.inputCollectionOfScalar("dirichlet_val", dirichlet_boundary);
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);
    auto computeInteriorFlux = [&](const auto& u) {
        return (-itrans * er.gradient(u));
    };
    auto computeBoundaryFlux = [&](const auto& u) {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary));
        const CollOfScalar dir_fluxes = ((er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary) * dir_sign) * (u_dirbdycells - dirichlet_val));
        return er.operatorExtend(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    auto computeResidual = [&](const auto& u) {
        const CollOfScalar ifluxes = computeInteriorFlux(u);
        const CollOfScalar bfluxes = computeBoundaryFlux(u);
        const CollOfScalar fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
        const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(fluxes)));
        return residual;
    };
    const CollOfScalar explicitu = (u0 - computeResidual(u0));
    const CollOfScalar u = er.newtonSolve(computeResidual, u0);
    er.output("explicitu", explicitu);
    er.output("u", u);

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
