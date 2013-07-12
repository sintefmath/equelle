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


typedef AutoDiff::ForwardBlock<double> ADB;
typedef ADB::V V;
typedef ADB::M M;
typedef Eigen::Array<double,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::RowMajor> DataBlock;


double norm(const ADB& x)
{
    return x.value().matrix().norm();
}



double euc_diff(const double* v0, const double* v1, const int dim)
{
    double l2 = 0.0;
    for (int dd = 0; dd < dim; ++dd) {
        const double diff = v0[dd] - v1[dd];
        l2 += diff*diff;
    }
    return std::sqrt(l2);
}




class ResidualComputer
{
public:
    ResidualComputer(const UnstructuredGrid& grid,
                     const Opm::parameter::ParameterGroup& param)
        : grid_(grid),
          ops_(grid),
          dt_(param.getDefault("dt", 0.5)),
          k_(param.getDefault("k", 0.3))
    {
    }

    ADB compute(const ADB& u,
                const V& u0)
    {
        const int nc = grid_.number_of_cells;
        // const int nf = grid_.number_of_faces;
        const int dim = grid_.dimensions;
        const int num_internal = ops_.internal_faces.size();

        // ============= Hopefully generated code starts here ================

        // --------------------------------------------------------------------------------
        // trans : scalar(faces(grid)) = k * area / length(centroid(first) - centroid(second))
        // --------------------------------------------------------------------------------
        // trans is a V and not an ADB since it does not depend on u.
        const V kvec = V::Constant(num_internal, k_);
        const V fareas = Eigen::Map<V>(grid_.face_areas, num_internal);
        // Centroids are points, in UnstructuredGrid, all is just one long array { x0, y0, z0, x1, y1, .... }.
#if 1
        const DataBlock cell_centroids = Eigen::Map<DataBlock>(grid_.cell_centroids, nc, dim);
        // #rows of face_centroid_diffs will be equal to number of inner faces. Will have dim columns.
        const DataBlock face_centroid_diffs = ops_.ngrad * cell_centroids.matrix();
        const V centroid_diff_lengths = face_centroid_diffs.matrix().rowwise().norm();
#else
        // Won't work since c0 or c1 can be -1 (on the boundary)
        V centroid_diff_lengths = V::Zero(nf);
        for (int f = 0; f < nf; ++f) {
            const int c0 = grid_.face_cells[2*f];
            const int c1 = grid_.face_cells[2*f+1];
            centroid_diff_lengths[f] = euc_diff(grid_.cell_centroids + dim*c0,
                                                grid_.cell_centroids + dim*c1, dim);
        }
#endif
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




ADB solve(const Opm::LinearSolverInterface& linsolver, const ADB& residual, const ADB& u)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr = residual.derivative()[0];

    V du = V::Zero(residual.size());
    Opm::LinearSolverInterface::LinearSolverReport rep
        = linsolver.solve(matr.rows(), matr.nonZeros(),
                          matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                          residual.value().data(), du.data());
    if (!rep.converged) {
        THROW("Linear solver convergence failure.");
    }
    return u - du;
}



int main(int argc, char** argv)
{
    // Read user parameters.
    // dt, k
    Opm::parameter::ParameterGroup param(argc, argv, false);

    const int n = param.getDefault("n", 6);

    Opm::LinearSolverFactory linsolver(param);

    // Obtain grid.
    // Create a 1d cartesian grid.
    Opm::GridManager gm(n, 1);
    const UnstructuredGrid& grid = *gm.c_grid();

    // Initial values. Must be worked on...
    V u0 = V::Zero(n);
    u0[0] = 0.5;
    u0[n-1] = 1.5;

    // Create unknowns.
    ADB u = ADB::variable(0, u0, { n } );

    ResidualComputer rescomp(grid, param);

    // Set up Newton loop.
    ADB residual = rescomp.compute(u, u0);
    const int max_iter = 10;
    const double tol = 1e-6;
    int iter = 0;
    while (norm(residual) > tol && iter < max_iter) {
        u = solve(linsolver, residual, u);
        residual = rescomp.compute(u, u0);
        std::copy(u.value().data(), u.value().data() + n, std::ostream_iterator<double>(std::cout, " "));
        std::cout << std::endl;
    }
}
