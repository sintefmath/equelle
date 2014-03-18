
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

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;

void ensureRequirements(const EquelleRuntimeCUDA& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCUDA er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const Scalar k = er.inputScalarWithDefault("k", double(0.3));
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    const CollOfScalar u0 = er.inputCollectionOfScalar("u0", er.allCells());
    const CollOfFace dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const CollOfScalar dirichlet_val = er.inputCollectionOfScalar("dirichlet_val", dirichlet_boundary);
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    std::cout << "-----------Before itrans-------------\n";
    
    std::cout << "-------Debug section:\n";
    std::cout << "-------first\n";
    const CollOfVector centroid_first = er.centroid(first);
    std::cout << "-------first: Dim = " << centroid_first.dim() << " and size " << centroid_first.size() << "\n";
    std::cout << "-------second\n";
    const CollOfVector centroid_second = er.centroid(second);
    std::cout << "-------second: Dim = " << centroid_second.dim() << " and size " << centroid_second.size() << "\n";
    
    std::cout << "-------first_const\n";
    const CollOfVector centroid_first_const = centroid_first;
    std::cout << "-------first_const: Dim = " << centroid_first_const.dim() << " and size " << centroid_first_const.size() << "\n";
    std::cout << "-------second_const\n";
    const CollOfVector centroid_second_const = centroid_second;
    std::cout << "-------second_const: Dim = " << centroid_second_const.dim() << " and size " << centroid_second_const.size() << "\n";
    std::cout << "-------second: Dim = " << centroid_second.dim() << " and size " << centroid_second.size() << "\n";

    std::cout << "-------diff\n";
    const CollOfVector centroid_diff = centroid_first - centroid_second;
    std::cout << "-------diff: Dim = " << centroid_diff.dim() << " and size " << centroid_diff.size() << "\n";
    std::cout << "-------norm(diff)\n";
    const CollOfScalar centroid_diff_norm = er.norm(centroid_diff);
    std::cout << "-------norm(int_faces)\n";
    const CollOfScalar norm_int_faces = er.norm(interior_faces);
    norm_int_faces.debug();
    std::cout << "-------division\n";
    const CollOfScalar scaled_itrans = norm_int_faces/centroid_diff_norm;
    scaled_itrans.debug();
    std::cout << "-------Debug section finished\n";

    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    itrans.debug();
    std::cout << "-----------After itrans--------------\n";
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);
    std::function<CollOfScalar(const CollOfScalar&)> computeInteriorFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    std::function<CollOfScalar(const CollOfScalar&)> computeBoundaryFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary));
        const CollOfScalar dir_fluxes = ((er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary) * dir_sign) * (u_dirbdycells - dirichlet_val));
        return er.operatorExtend(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    CollOfScalar expU;
    expU = u0;
    for (const Scalar& dt : timesteps) {
        std::function<CollOfScalar(const CollOfScalar&)> computeResidual = [&](const CollOfScalar& u) -> CollOfScalar {
            const CollOfScalar ifluxes = computeInteriorFlux(u);
            const CollOfScalar bfluxes = computeBoundaryFlux(u);
            const CollOfScalar fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
            const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(fluxes)));
            return residual;
        };
        expU = (expU - computeResidual(expU));
        er.output("expU", expU);
    }
    er.output("expU", expU);

    std::cout << "------ End of program ------\n";

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}
