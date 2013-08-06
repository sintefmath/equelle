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
    // k : Collection Of Scalar On AllCells() = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
    // Scalars k // This would make 'k' contain separate values for all cells.
    // k : Scalar = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
    double k;
    // dt : Scalar = UserSpecifiedScalarWithDefault(0.5) # Time step length.
    double dt;
    // u0 : Collection Of Scalar On AllCells() = UserSpecifiedCollectionOfScalar( AllCells() )
    CollOfScalars u0;

    // ============= Generated code ends here ================

    UserParameters(const Opm::parameter::ParameterGroup& param,
                   const EquelleRuntimeCPU& er)
    {
        // ============= Generated code starts here ================
        // k : Scalar = UserSpecifiedScalarWithDefault(0.3) # Heat diffusion constant.
        k = param.getDefault("k", 0.3);
        // dt : Scalar = UserSpecifiedScalarWithDefault(0.5) # Time step length.
        dt = param.getDefault("dt", 0.5);
        // u0 : Collection Of Scalar On AllCells() = UserSpecifiedCollectionOfScalar( AllCells() )
        u0 = er.getUserSpecifiedCollectionOfScalar(param, "u0", er.allCells().size());
        // ============= Generated code ends here ================
    }
};





class ResidualComputer : public ResidualComputerInterface
{
public:
    /// Initialization.
    ResidualComputer(const EquelleRuntimeCPU& er, const UserParameters& up)
        : er_(er), up_(up)
    {
    }

    // Compute the (possibly nonlinear) residual with derivative information.
    // This is the most important generated function.
    CollOfScalarsAD compute(const CollOfScalarsAD& u) const
    {
        // ============= Generated code starts here ================

        // --------------------------------------------------------------------------------
        // vol = Volume( AllCells() )    # Deduced type: Scalar On AllCells()
        // --------------------------------------------------------------------------------
        const CollOfScalars vol = er_.volume( er_.allCells() );

        // --------------------------------------------------------------------------------
	// interior_faces = InteriorFaces()       # Deduced type:  Collection Of Face Subset Of AllFaces()
	// first = FirstCell(interior_faces)      # Deduced type:  Collection Of Cell On interior_faces
	//                                        # Equivalent to: Collection Of Cell On InteriorFaces()
	// second = SecondCell(interior_faces)    # Deduced type:  Same as for 'first'.
        // --------------------------------------------------------------------------------
        const CollOfFaces interior_faces = er_.interiorFaces();
        const CollOfCells first = er_.firstCell( interior_faces );
        const CollOfCells second = er_.secondCell( interior_faces );

        // --------------------------------------------------------------------------------
        // trans : Collection Of Scalar On interior_faces = k * Area(interior_faces) / Length(Centroid(first) - Centroid(second))
        //    # Deduced (and declared) type: Collection Of Scalar On interior_faces
        // --------------------------------------------------------------------------------
        // trans is a CollOfScalars and not a CollOfScalarsAD since it does not depend on u.
        const CollOfScalars trans = up_.k * er_.area(interior_faces) / er_.length(er_.centroid(first) - er_.centroid(second));

        // --------------------------------------------------------------------------------
        // fluxes : Collection Of Scalar On interior_faces = - trans * Gradient(u)
        //    # Deduced (and declared) type: Scalar On interior_faces
        // --------------------------------------------------------------------------------
        // fluxes is a CollOfScalarsAD since it depends on u.
        const CollOfScalarsAD fluxes = trans * (er_.negGradient(u));

        // --------------------------------------------------------------------------------
        // residual : Collection Of Scalar On AllCells() = u - u0 + (dt / vol) * Divergence(fluxes) 
        //    # Deduced (and declared) type: Scalar On AllCells()
        // --------------------------------------------------------------------------------
        // residual is a CollOfScalarsAD since it depends on u and fluxes.
        const CollOfScalarsAD residual = u - up_.u0 + (up_.dt / vol) * (er_.divergence(fluxes));

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
    // u : Collection Of Scalar On AllCells() = u0
    // --------------------------------------------------------------------------------
    CollOfScalarsAD u = er.singlePrimaryVariable(up.u0);

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
    er.output("Final u is equal to: ", u);

    // ============= Generated code ends here ================

    return 0;
}




















