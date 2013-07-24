/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

// Obtain grid.
// Create a 1d cartesian grid.

// Read user parameters.
// dt, k

// Set up Newton loop.
// - convergence criteria
// - assemble linear system
//    - set up AD variables (since we know the primary unknown)
//    - arithmetic computation of residual (and Jacobian from AD)
//         -----> This is the important part!
// - solve linear system
// - update solution

// Output solution
// u

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

#include "common.hpp"




class ResidualComputer
{

public:


    ResidualComputer(const UnstructuredGrid& grid, const Opm::parameter::ParameterGroup& param)
        : grid_(grid),
          ops_(grid),
          dt_(param.getDefault("dt", 0.5)),
          k_(param.getDefault("k", 0.3))
    {
    }


    // @jny Tried to change the type of u0 from V to ADB, to simplify the "ping-ponging" of variables u and u0 in the Newton loop,
    //      but this didn't work as expected. Don't quite see why it shouldn't. (Some sort of AD-info dragged along from the u0 when
    //      it shouldn't have been?)
    ADB compute(const ADB& u, const V &u0)
    {
        const int nc = grid_.number_of_cells;
        const int dim = grid_.dimensions;
        const int num_internal = ops_.internal_faces.size();

        // ============= Hopefully generated code starts here ================

        // --------------------------------------------------------------------------------
        // trans : scalar(faces(grid)) = k * area / length(centroid(first) - centroid(second))
        // --------------------------------------------------------------------------------
        // trans is a V and not an ADB since it does not depend on u.
        const V kvec = V::Constant(num_internal, k_);
        const V fareas = Eigen::Map<V>(grid_.face_areas, num_internal);

        // -----------

        // Our original "alt. 6" version
        const DataBlock cell_centroids = Eigen::Map<DataBlock>(grid_.cell_centroids, nc, dim); // nc=number of cells, dim=2, but it's 1D since it is of size (nc, 1)
        const DataBlock face_centroid_diffs = ops_.ngrad * cell_centroids.matrix(); // We get "first centroid" - "second centroid" this way. (Second column to be discarded!)

        // A bit more explicit, but maybe more true to the Equelle-code
        const M first_faces  =   pick_elements(ops_.ngrad, 1);
        const M second_faces =   pick_elements(ops_.ngrad, -1);
        const DataBlock cell_centroids2 = Eigen::Map<DataBlock>(grid_.cell_centroids, nc, dim);
        const DataBlock first_centroids  = first_faces  * cell_centroids2.matrix();
        const DataBlock second_centroids = second_faces * cell_centroids2.matrix();
        const DataBlock face_centroid_diffs2 = first_centroids + second_centroids; // Note the sign, we get the "-" from the 'ngrad' above...

        // -----------

        const V centroid_diff_lengths = face_centroid_diffs.matrix().rowwise().norm(); // @jny Works because (x^2+y^2)^(1/2) == |x| for y==0?
        const V trans = kvec * fareas / centroid_diff_lengths;

        // --------------------------------------------------------------------------------
        // fluxes : scalar(faces(grid)) = - trans * grad(u)
        // --------------------------------------------------------------------------------
        const ADB fluxes = trans * (ops_.ngrad * u);

        // --------------------------------------------------------------------------------
        // vol : scalar(cells(grid)) = volume
        // --------------------------------------------------------------------------------
        const V vol = Eigen::Map<V>(grid_.cell_volumes, nc);

        // --------------------------------------------------------------------------------
        // residual : scalar(cells(grid)) = u - u0 + (dt / vol) * div(fluxes)
        // --------------------------------------------------------------------------------
        const ADB residual = u - u0 + (dt_ / vol) * (ops_.div * fluxes);

        // ============= Hopefully generated code ends here ================

        return residual;
    }


private:

    const UnstructuredGrid& grid_;
    HelperOps ops_;
    double dt_;
    double k_;

};




int main(int argc, char** argv)
{
    // Read user parameters. (dt, k, n)
    Opm::parameter::ParameterGroup param(argc, argv, false);
    const int n = param.getDefault("n", 6);
    Opm::LinearSolverFactory linsolver(param);

    // Create a 1D Cartesian grid.
    Opm::GridManager gm(n, 1);
    const UnstructuredGrid& grid = *gm.c_grid();

    // ============= More generated code starts here ================

    // --------------------------------------------------------------------------------
    // u0 : scalar(cells(grid))
    // --------------------------------------------------------------------------------
    V u0_vec = V::Zero( grid.number_of_cells );
    // Initial values. Must be worked on...
    u0_vec[0] = 0.5;
    u0_vec[grid.number_of_cells-1] = 1.5;

    // ADB u0 = ADB::variable(0, u0_vec, { grid.number_of_cells } ); // @jny Why didn't this work?
    V u0 = u0_vec;

    // Create unknowns.
    // --------------------------------------------------------------------------------
    // u : scalar(cells(grid)) = u0
    // --------------------------------------------------------------------------------
    ADB u = ADB::variable(0, u0, { grid.number_of_cells } ); // (block index, initialized from, and block structure)
    // ADB u = u0; // @jny Why didn't this work, for u0 of ADB-type?

    // ============= Generated code ends here ================

    ResidualComputer rescomp(grid, param);

    std::cout << std::endl;
    print_vec("Initial u:\t\t", u);

    // Set up Newton loop.
    ADB residual = rescomp.compute(u, u0); // ============= Generated code in here ================
    print_vec("Initial residual:\t", residual);

    const int max_iter = 10;
    const double tol = 1e-6;

    // ============= More generated code starts here ================

    // --------------------------------------------------------------------------------
    // newtonsolve(residual, u)
    // --------------------------------------------------------------------------------

    // Maybe a little of a stretch to call this generated code, but...

    int iter = 0;
    while ( (norm(residual)>tol) && (iter<max_iter) ) {

        std::cout << "\niter = " << iter << " (max = " << max_iter << "), norm(residual) = " << norm(residual) << " (tol = " << tol << ")" << std::endl;

#if 0
        // @jny Was this loop body correct? I.e., correct to use u0 in the residual computation every time?!
        u = old_solve(linsolver, residual, u);
        residual = rescomp.compute(u, u0); // ============= Generated code is called here ================
#else
        // @jny Or is this better? (Of course, no apparent change for a linear problem...)
        const V du = solve(linsolver, residual);
        const ADB new_u = u - du;
        // @jny If rescomp.compute could take and ADB as type for the second parameter, we could just call it directly now,
        //      but this seems not to work, I don't quite see why.
        {
            // "Casting" u from V to ADB in order to call rescomp.compute with u as second parameter
            V uv = V::Zero( u.size() );
            for (int i=0; i<u.size(); i++)
                uv[i] = u.value().data()[i];
            residual = rescomp.compute(new_u, uv); // ============= Generated code in here ================
        }
        // And this step was the one missing from the section above:
        u = new_u;
#endif

        print_vec("\tu:\t", u);
        print_vec("\tresidual:\t", residual);

    }

    // ============= Generated code ends here ================

    return 0;
}




































