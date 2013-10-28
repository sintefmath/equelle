/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

////////////////////////////////////////////////////////////////////////
// NOTE
// This file is written to be similar to the intended output from the
// Equelle compiler. It is not actually output from the compiler,
// although it has beem modified to better match the compiler output
// as that has changed.
////////////////////////////////////////////////////////////////////////

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
    // k : Scalar = UserSpecifiedScalarWithDefault("k", 0.3) # Heat diffusion constant.
    // dt : Scalar = UserSpecifiedScalarWithDefault("dt", 0.5) # Time step length.
    // u0 : Collection Of Scalar On AllCells()
    // u0 = UserSpecifiedCollectionOfScalar("u0", AllCells())
    // dirichlet_boundary : Collection Of Face On BoundaryFaces()
    // dirichlet_boundary = UserSpecifiedCollectionOfFace( BoundaryFaces() )
    // dirichlet_val : Collection Of Scalar On dirichlet_boundary
    // dirichlet_val = UserSpecifiedCollectionOfScalar("dirichlet_val", dirichlet_boundary)
    // --------------------------------------------------------------------------------
    const Scalar k = er.userSpecifiedScalarWithDefault("k", double(0.3));
    const Scalar dt = er.userSpecifiedScalarWithDefault("dt", double(0.5));
    const CollOfScalar u0 = er.userSpecifiedCollectionOfScalar("u0", er.allCells());
    const CollOfFace dirichlet_boundary = er.userSpecifiedCollectionOfFaceSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const CollOfScalar dirichlet_val = er.userSpecifiedCollectionOfScalar("dirichlet_val", dirichlet_boundary);

    // --------------------------------------------------------------------------------
    // # Compute interior transmissibilities.
    // vol = |AllCells()|                                         # Deduced type:  Collection Of Scalar On AllCells()
    // interior_faces = InteriorFaces()                           # Deduced type:  Collection Of Face
    // first = FirstCell(interior_faces)                          # Deduced type:  Collection Of Cell On interior_faces
    // 							          # Equivalent to: Collection Of Cell On InteriorFaces()
    // second = SecondCell(interior_faces)                        # Deduced type:  Same as for 'first'.
    // itrans : Collection Of Scalar On interior_faces = k * |interior_faces| / |Centroid(first) - Centroid(second)|
    // --------------------------------------------------------------------------------
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));

    // --------------------------------------------------------------------------------
    // # Compute boundary transmissibilities.
    // bf = BoundaryFaces()
    // bf_cells = IsEmpty(FirstCell(bf)) ? SecondCell(bf) : FirstCell(bf)
    // bf_sign = IsEmpty(FirstCell(bf)) ? (-1 On bf) : (1 On bf)
    // btrans = k * |bf| / |Centroid(bf) - Centroid(bf_cells)|
    // --------------------------------------------------------------------------------
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorOn(-double(1), bf), er.operatorOn(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));

    // --------------------------------------------------------------------------------
    // # Compute quantities needed for boundary conditions.
    // dir_sign = bf_sign On dirichlet_boundary
    // --------------------------------------------------------------------------------
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);

    // --------------------------------------------------------------------------------
    // # Compute flux for interior faces.
    // computeInteriorFlux : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On InteriorFaces()
    // computeInteriorFlux(u) = {
    //     -> -itrans * Gradient(u)
    // }
    // --------------------------------------------------------------------------------
    auto computeInteriorFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };

    // --------------------------------------------------------------------------------
    // # Compute flux for boundary faces.
    // computeBoundaryFlux : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On BoundaryFaces()
    // computeBoundaryFlux(u) = {
    //     # Compute flux at Dirichlet boundaries.
    //     u_dirbdycells = u On (bf_cells On dirichlet_boundary)
    //     dir_fluxes = (btrans on dirichlet_boundary) * dir_sign * (u_dirbdycells - dirichlet_val)
    //     # Extending with zero away from Dirichlet boundaries (i.e. assuming no-flow elsewhere).
    //     -> dir_fluxes On BoundaryFaces()
    // }
    // --------------------------------------------------------------------------------
    auto computeBoundaryFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary));
        const CollOfScalar dir_fluxes = ((er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary) * dir_sign) * (u_dirbdycells - dirichlet_val));
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
    //     -> residual
    // }
    // --------------------------------------------------------------------------------
    auto computeResidual = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar ifluxes = computeInteriorFlux(u);
        const CollOfScalar bfluxes = computeBoundaryFlux(u);
        const CollOfScalar fluxes = (er.operatorOn(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorOn(bfluxes, er.boundaryFaces(), er.allFaces()));
        const CollOfScalar residual = ((u - u0) + ((dt / vol) * er.divergence(fluxes)));
        return residual;
    };

    // --------------------------------------------------------------------------------
    // explicitu = u0 - computeResidual(u0)
    // u = NewtonSolve(computeResidual, u0)
    // --------------------------------------------------------------------------------
    const CollOfScalar explicitu = (u0 - computeResidual(u0));
    const CollOfScalar u = er.newtonSolve(computeResidual, u0);

    // --------------------------------------------------------------------------------
    // Output("explicitu", explicitu)
    // Output("u", u)
    // --------------------------------------------------------------------------------
    er.output("explicitu", explicitu);
    er.output("u", u);

    // ============= Generated code ends here ================

    return 0;
}
