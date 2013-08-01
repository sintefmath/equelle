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




struct UserParameters
{
    // ============= Generated code starts here ================
    // k : Scalar = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
    double k;
    // dt : Scalar = UserSpecifiedScalarWithDefault(0.5) # Time step length.
    double dt;
    // u0 : Scalar On Cells(Grid) = UserSpecifiedScalars
    Scalars u0;
    // ============= Generated code ends here ================

    UserParameters(const Opm::parameter::ParameterGroup& param,
                   const EquelleRuntimeCPU& er)
    {
        // ============= Generated code starts here ================
        // k : Scalar = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
        k = param.getDefault("k", 0.3);
        // dt : Scalar = UserSpecifiedScalarWithDefault(0.5) # Time step length.
        dt = param.getDefault("dt", 0.5);
        // u0 : Scalar On Cells(Grid) = UserSpecifiedScalars
        u0 = EquelleRuntimeCPU::getUserSpecifiedScalars(param, "u0", er.allCells().size());
        // ============= Generated code ends here ================
    }
};





class ResidualComputer : public ResidualComputerInterface
{
public:
    /// Initialization.
    ResidualComputer(const EquelleRuntimeCPU& er,
                     const UserParameters& up)
        : er_(er), up_(up)
    {
    }

    // Compute the (possibly nonlinear) residual with derivative information.
    // This is the most important generated function.
    ScalarsAD compute(const ScalarsAD& u) const
    {
        // ============= Generated code starts here ================

        // --------------------------------------------------------------------------------
        // vol = Volume(Cells(Grid))    # Deduced type: Scalar On Cells(Grid)
        // --------------------------------------------------------------------------------
        const Scalars vol = er_.volume(er_.allCells());

        // --------------------------------------------------------------------------------
        // internal_faces = InternalFaces(Grid)   # Deduced type: Face On InternalFaces(Grid)
        // first = FirstCell(internal_faces)      # Deduced type: Cell On internal_faces
        // second = SecondCell(internal_faces)    # Deduced type: Cell On internal_faces
        // --------------------------------------------------------------------------------
        const Faces internal_faces = er_.internalFaces();
        const Cells first = er_.firstCell(internal_faces);
        const Cells second = er_.secondCell(internal_faces);

        // --------------------------------------------------------------------------------
        // trans : Scalar On internal_faces = k * Area(internal_faces) / Length(Centroid(first) - Centroid(second))
        //    # Deduced (and declared) type: Scalar On internal_faces
        // --------------------------------------------------------------------------------
        // trans is a Scalars and not a ScalarsAD since it does not depend on u.
        const Scalars term1 = Scalars::Constant(internal_faces.size(), up_.k);
        const Scalars term2 = er_.area(internal_faces);
        const Scalars term3 = er_.length(er_.centroid(first) - er_.centroid(second));
        const Scalars trans = term1 * term2 / term3;
        // const Scalars trans = up_.k * er_.area(internal_faces) / er_.length(er_.centroid(first) - er_.centroid(second));

        // --------------------------------------------------------------------------------
        // fluxes : Scalar On internal_faces = - trans * Gradient(u)
        //    # Deduced (and declared) type: Scalar On internal_faces
        // --------------------------------------------------------------------------------
        // fluxes is a ScalarsAD since it depends on u.
        const ScalarsAD fluxes = trans * (er_.negGradient(u));

        // --------------------------------------------------------------------------------
        // residual : Scalar On Cells(Grid) = u - u0 + (dt / vol) * Divergence(fluxes) 
        //    # Deduced (and declared) type: Scalar On Cells(Grid)
        // --------------------------------------------------------------------------------
        // residual is a ScalarsAD since it depends on u and fluxes.
        const ScalarsAD residual = u - up_.u0 + (up_.dt / vol) * (er_.divergence(fluxes));

        // ============= Hopefully generated code ends here ================

        return residual;
    }

private:
    const EquelleRuntimeCPU& er_;
    const UserParameters& up_;
};





int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCPU er(param);

    // Obtain user parameters.
    UserParameters up(param, er);

    // ============= More generated code starts here ================

    // Create unknowns.
    // --------------------------------------------------------------------------------
    // u : Scalar On Cells(Grid) = u0
    // --------------------------------------------------------------------------------
    ScalarsAD u = EquelleRuntimeCPU::singlePrimaryVariable(up.u0);

    // --------------------------------------------------------------------------------
    // NewtonSolve(residual, u)
    // --------------------------------------------------------------------------------
    ResidualComputer rescomp(er, up);
    u = er.newtonSolve(rescomp, u);

    // --------------------------------------------------------------------------------
    // Output(u)
    // Output(fluxes)
    // --------------------------------------------------------------------------------
    // Not handling output of fluxes currently, since they are not in scope here.
    EquelleRuntimeCPU::output("Final u is equal to: ", u);

    // ============= Generated code ends here ================

    return 0;
}




















