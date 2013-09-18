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
    // --------------------------------------------------------------------------------
    const auto k = param.getDefault("k", 0.3);
    const auto dt = param.getDefault("dt", 0.5);
    const auto u0 = er.getUserSpecifiedCollectionOfScalar(param, "u0", er.allCells().size());

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
    // # Compute flux for interior faces.
    // computeInteriorFlux : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On InteriorFaces()
    // computeInteriorFlux(u) = {
    //     flux = -itrans * Gradient(u)
    //     return flux
    // }
    // --------------------------------------------------------------------------------
    auto computeInteriorFlux = [&](const CollOfScalars u) -> CollOfScalars {
        const CollOfScalars flux = -itrans * er.gradient(u);
        return flux;
    };
    // auto computeInteriorFluxAD = [&](const CollOfScalarsAD u) -> CollOfScalarsAD {
    //     const CollOfScalarsAD flux = -itrans * er.gradient(u);
    //     return flux;
    // };

    // --------------------------------------------------------------------------------
    // # Compute the residual for the heat equation.
    // computeResidual : Function(u : Collection Of Scalar On AllCells()) -> Collection Of Scalar On AllCells()
    // computeResidual(u) = {
    //     ifluxes = computeInteriorFlux(u)
    //     # Deduced type: Collection Of Scalar On AllCells()
    //     residual = u - u0 + (dt / vol) * Divergence(ifluxes)
    //     return residual
    // }
    // --------------------------------------------------------------------------------
    auto computeResidual = [&](const CollOfScalars u) -> CollOfScalars {
        const CollOfScalars ifluxes = computeInteriorFlux(u);
        const CollOfScalars residual = u - u0 + (dt / vol) * er.divergence(ifluxes);
        return residual;
    };
    // auto computeResidualAD = [&](const CollOfScalarsAD u) -> CollOfScalarsAD {
    //     const CollOfScalarsAD ifluxes = computeInteriorFluxAD(u);
    //     const CollOfScalarsAD residual = u - u0 + (dt / vol) * er.divergence(ifluxes);
    //     return residual;
    // };

    // --------------------------------------------------------------------------------
    // explicitu = u0 - computeResidual(u0)
    // u = NewtonSolve(computeResidual, u0)
    // --------------------------------------------------------------------------------
    const CollOfScalars explicitu = u0 - computeResidual(u0);
    // const CollOfScalarsAD u = er.newtonSolve(computeResidualAD, u0);

    // --------------------------------------------------------------------------------
    // Output(explixitu)
    // Output(u)
    // --------------------------------------------------------------------------------
    er.output("explicitu", explicitu);
    // er.output("u", u);

    // ============= Generated code ends here ================


    return 0;
}




















