/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

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

    // --------------------------------------------------------------------------------
    // k : Scalar = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
    // dt : Scalar = UserSpecifiedScalarWithDefault(0.5) # Time step length.
    // u0 : Collection Of Scalar On AllCells() = UserSpecifiedCollectionOfScalar( AllCells() )
    // dirichlet_boundary : Collection Of Face On BoundaryFaces() = UserSpecifiedCollectionOfFace( BoundaryFaces() )
    // dirichlet_val : Collection Of Scalar On dirichlet_boundary = UserSpecifiedCollectionOfScalar( dirichlet_boundary )
    // --------------------------------------------------------------------------------
    const auto k = param.getDefault("k", 0.3);
    const auto dt = param.getDefault("dt", 0.5);
    const auto u0 = er.getUserSpecifiedCollectionOfScalar(param, "u0", er.allCells().size());
    const auto dirichlet_boundary = er.getUserSpecifiedCollectionOfFaceSubsetOf(param, "dirichlet_boundary", er.boundaryFaces());
    const auto dirichlet_val = er.getUserSpecifiedCollectionOfScalar(param, "dirichlet_val", dirichlet_boundary.size());

    // --------------------------------------------------------------------------------
    // # Compute interior transmissibilities.
    // vol = |AllCells()|                                         # Deduced type:  Collection Of Scalar On AllCells()
    // interior_faces = InteriorFaces()                           # Deduced type:  Collection Of Face
    // first = FirstCell(interior_faces)                          # Deduced type:  Collection Of Cell On interior_faces
    // 							          # Equivalent to: Collection Of Cell On InteriorFaces()
    // second = SecondCell(interior_faces)                        # Deduced type:  Same as for 'first'.
    // itrans : Collection Of Scalar On interior_faces = k * |interior_faces| / |Centroid(first) - Centroid(second)|
    // --------------------------------------------------------------------------------
    const auto vol = er.norm(er.allCells());
    const auto interior_faces = er.interiorFaces();
    const auto first = er.firstCell(interior_faces);
    const auto second = er.secondCell(interior_faces);
    const CollOfScalars itrans = k * er.norm(interior_faces) / er.norm(er.centroid(first) - er.centroid(second));

    // --------------------------------------------------------------------------------
    // # Compute boundary transmissibilities.
    // bf = BoundaryFaces()
    // bf_cells = IsEmpty(FirstCell(bf)) ? SecondCell(bf) : FirstCell(bf)
    // bf_sign = IsEmpty(FirstCell(bf))) ? (-1 On bf_cells) : (1 On bf_cells)
    // btrans = k * |bf| / |Centroid(bf) - Centroid(bf_cells)|
    // --------------------------------------------------------------------------------
    const auto bf = er.boundaryFaces();
    const auto bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const auto bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorOn(double(-1), bf_cells), er.operatorOn(double(1), bf_cells));
    const CollOfScalars btrans = k * er.norm(bf) / er.norm(er.centroid(bf) - er.centroid(bf_cells));

    // --------------------------------------------------------------------------------
    // # Compute quantities needed for boundary conditions.
    // dir_sign = bf_sign On dirichlet_boundary
    // --------------------------------------------------------------------------------
    const auto dir_sign = er.operatorOn(bf_sign, bf, dirichlet_boundary);

    // --------------------------------------------------------------------------------
    // # Compute flux for interior faces.
    // computeInteriorFlux : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On InteriorFaces()
    // computeInteriorFlux(u) = {
    //     return -itrans * Gradient(u)
    // }
    // --------------------------------------------------------------------------------
    auto computeInteriorFlux = [&](const CollOfScalars u) -> CollOfScalars {
        return -itrans * er.gradient(u);
    };
    auto computeInteriorFluxAD = [&](const CollOfScalarsAD u) -> CollOfScalarsAD {
        return -itrans * er.gradient(u);
    };

    // --------------------------------------------------------------------------------
    // # Compute flux for boundary faces.
    // computeBoundaryFlux : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On BoundaryFaces()
    // computeBoundaryFlux(u) = {
    //     # Compute flux at Dirichlet boundaries.
    //     dir_fluxes = btrans * dir_sign * (u_dirbdycells - dirichlet_val)
    //     # Extending with zero away from Dirichlet boundaries (i.e. assuming no-flow elsewhere).
    //     return dir_fluxes On BoundaryFaces()
    // }
    // --------------------------------------------------------------------------------
    auto computeBoundaryFlux = [&](const CollOfScalars u) -> CollOfScalars {
        const CollOfScalars u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, bf, dirichlet_boundary));
        const CollOfScalars dir_fluxes = btrans * dir_sign * (u_dirbdycells - dirichlet_val);
        return er.operatorOn(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    auto computeBoundaryFluxAD = [&](const CollOfScalarsAD u) -> CollOfScalarsAD {
        const CollOfScalarsAD u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, bf, dirichlet_boundary));
        const CollOfScalarsAD dir_fluxes = btrans * dir_sign * (u_dirbdycells - dirichlet_val);
        return er.operatorOn(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };

    // --------------------------------------------------------------------------------
    // # Compute the residual for the heat equation.
    // computeResidual : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On AllCells()
    // computeResidual(u) = {
    //     ifluxes = computeInteriorFlux(u)
    //     bfluxes = computeBoundaryFlux(u)
    //     # Extend both ifluxes and bfluxes to AllFaces() and add to get all fluxes.
    //     fluxes = (ifluxes On AllFaces()) + (bfluxes On AllFaces())
    //     # Deduced type: Collection Of Scalar On AllCells()
    //     residual = u - u0 + (dt / vol) * Divergence(fluxes)
    //     return residual
    // }
    // --------------------------------------------------------------------------------
    auto computeResidual = [&](const CollOfScalars u) -> CollOfScalars {
        const CollOfScalars ifluxes = computeInteriorFlux(u);
        const CollOfScalars bfluxes = computeBoundaryFlux(u);
        const CollOfScalars fluxes = er.operatorOn(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorOn(bfluxes, er.boundaryFaces(), er.allFaces());
        const CollOfScalars residual = u - u0 + (dt / vol) * er.divergence(fluxes);
        return residual;
    };
    auto computeResidualAD = [&](const CollOfScalarsAD u) -> CollOfScalarsAD {
        const auto ifluxes = computeInteriorFluxAD(u);
        const auto bfluxes = computeBoundaryFluxAD(u);
        const auto fluxes = er.operatorOn(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorOn(bfluxes, er.boundaryFaces(), er.allFaces());
        const CollOfScalarsAD residual = u - u0 + (dt / vol) * er.divergence(fluxes);
        return residual;
    };

    // --------------------------------------------------------------------------------
    // explicitu = u0 - computeResidual(u0)
    // u = NewtonSolve(computeResidual, u0)
    // --------------------------------------------------------------------------------
    const CollOfScalars explicitu = u0 - computeResidual(u0);
    const CollOfScalarsAD u = er.newtonSolve(computeResidualAD, u0);

    // --------------------------------------------------------------------------------
    // Output(u)
    // Output(fluxes)
    // --------------------------------------------------------------------------------
    // Not handling output of fluxes currently, since they are not in scope here.
    er.output("Explicit u is equal to: ", explicitu);
    er.output("Implicit u is equal to: ", u);

    // ============= Generated code ends here ================

    return 0;
}




















