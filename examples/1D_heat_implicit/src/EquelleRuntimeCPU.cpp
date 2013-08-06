/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeCPU.hpp"
#include <iomanip>
#include <fstream>
#include <iterator>


EquelleRuntimeCPU::EquelleRuntimeCPU(const Opm::parameter::ParameterGroup& param)
    : grid_manager_(param.getDefault("n", 6), 1),
      grid_(*grid_manager_.c_grid()),
      ops_(grid_),
      linsolver_(param)
{
}


Cells EquelleRuntimeCPU::allCells() const
{
    const int nc = grid_.number_of_cells;
    Cells cells(nc);
    for (int c = 0; c < nc; ++c) {
        cells[c].index = c;
    }
    return cells;
}


Faces EquelleRuntimeCPU::allFaces() const
{
    const int nf = grid_.number_of_faces;
    Faces faces(nf);
    for (int f = 0; f < nf; ++f) {
        faces[f].index = f;
    }
    return faces;
}


Faces EquelleRuntimeCPU::interiorFaces() const
{
    const int nif = ops_.internal_faces.size();
    Faces ifaces(nif);
    for (int i = 0; i < nif; ++i) {
        ifaces[i].index = ops_.internal_faces(i);
    }
    return ifaces;
}


Cells EquelleRuntimeCPU::firstCell(const Faces& faces) const
{
    const int n = faces.size();
    Cells fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index];
    }
    return fcells;
}


Cells EquelleRuntimeCPU::secondCell(const Faces& faces) const
{
    const int n = faces.size();
    Cells fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index + 1];
    }
    return fcells;
}



CollOfScalars EquelleRuntimeCPU::area(const Faces& faces) const
{
    const int n = faces.size();
    CollOfScalars areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = grid_.face_areas[faces[i].index];
    }
    return areas;
}


CollOfScalars EquelleRuntimeCPU::volume(const Cells& cells) const
{
    const int n = cells.size();
    CollOfScalars volumes(n);
    for (int i = 0; i < n; ++i) {
        volumes[i] = grid_.cell_volumes[cells[i].index];
    }
    return volumes;
}


CollOfScalars EquelleRuntimeCPU::length(const CollOfVectors& vectors) const
{
    return vectors.matrix().rowwise().norm();
}


CollOfVectors EquelleRuntimeCPU::centroid(const Faces& faces) const
{
    const int n = faces.size();
    const int dim = grid_.dimensions;
    CollOfVectors centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.face_centroids + dim * faces[i].index;
        for (int d = 0; d < dim; ++d) {
            centroids(i, d) = fc[d];
        }
    }
    return centroids;
}


CollOfVectors EquelleRuntimeCPU::centroid(const Cells& cells) const
{
    const int n = cells.size();
    const int dim = grid_.dimensions;
    CollOfVectors centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.cell_centroids + dim * cells[i].index;
        for (int d = 0; d < dim; ++d) {
            centroids(i, d) = fc[d];
        }
    }
    return centroids;
}

CollOfScalarsAD EquelleRuntimeCPU::negGradient(const CollOfScalarsAD& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield;
}


CollOfScalarsAD EquelleRuntimeCPU::divergence(const CollOfScalarsAD& face_fluxes) const
{
    return ops_.div * face_fluxes;
}


CollOfScalars EquelleRuntimeCPU::solveForUpdate(const CollOfScalarsAD& residual) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr = residual.derivative()[0];

    CollOfScalars du = CollOfScalars::Zero(residual.size());

    // solve(n, # nonzero values ("val"), ptr to col indices
    // ("col_ind"), ptr to row locations in val array ("row_ind")
    // (these two may be swapped, not sure about the naming convention
    // here...), array of actual values ("val") (I guess... '*sa'...),
    // rhs, solution)
    Opm::LinearSolverInterface::LinearSolverReport rep
        = linsolver_.solve(matr.rows(), matr.nonZeros(),
                           matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                           residual.value().data(), du.data());
    if (!rep.converged) {
        THROW("Linear solver convergence failure.");
    }
    return du;
}


CollOfScalarsAD EquelleRuntimeCPU::newtonSolve(const ResidualComputerInterface& rescomp,
                                         const CollOfScalarsAD& u_initialguess) const
{

    // Set up Newton loop.
    CollOfScalarsAD u = u_initialguess;
    output("Initial u:\t\t", u);
    output("\tnorm = ", norm(u));
    CollOfScalarsAD residual = rescomp.compute(u); //  Generated code in here
    output("Initial residual:\t", residual);
    output("\tnorm = ", norm(residual));
    const int max_iter = 10;
    const double tol = 1e-6;
    int iter = 0;

    // Execute newton loop until residual is small or we have used too many iterations.
    while ( (norm(residual) > tol) && (iter < max_iter) ) {
        // Debugging output not specified in Equelle.
        std::cout << "\niter = " << iter << " (max = " << max_iter
                  << "), norm(residual) = " << norm(residual)
                  << " (tol = " << tol << ")" << std::endl;

        // Solve linear equations for du, apply update.
        const CollOfScalars du = solveForUpdate(residual);
        u = u - du;

        // Recompute residual.
        residual = rescomp.compute(u);

        // Debugging output not specified in Equelle.
        output("\tu:\t", u);
        output("\tnorm = ", norm(u));
        output("\tresidual:\t", residual);
        output("\tnorm = ", norm(residual));

        ++iter;
    }
    return u;
}


double EquelleRuntimeCPU::norm(const CollOfScalars& vals) const
{
    return vals.matrix().norm();
}


double EquelleRuntimeCPU::norm(const CollOfScalarsAD& vals) const
{
    return norm(vals.value());
}


void EquelleRuntimeCPU::output(const std::string& tag, const double val)
{
    std::cout << tag << val << std::endl;
}


void EquelleRuntimeCPU::output(const std::string& tag, const CollOfScalars& vals)
{
    std::cout << tag;
    for (int i = 0; i < vals.size(); ++i) {
        std::cout << std::setw(15) << std::left << ( vals[i] ) << " ";
    }
    std::cout << std::endl;
}


void EquelleRuntimeCPU::output(const std::string& tag, const CollOfScalarsAD& vals)
{
    output(tag, vals.value());
}


CollOfScalars EquelleRuntimeCPU::getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
							      const std::string& name,
							      const int size)
{
    const bool from_file = param.getDefault(name + "_from_file", false);
    if (from_file) {
        const std::string filename = param.get<std::string>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            THROW("Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;
        std::vector<double> data(beg, end);
        if (int(data.size()) != size) {
            THROW("Unexpected size of input data for " << name << " in file " << filename);
        }
        return CollOfScalars(Eigen::Map<CollOfScalars>(&data[0], size));
    } else {
        // Uniform values.
        return CollOfScalars::Constant(size, param.get<double>(name));
    }
}


CollOfScalarsAD EquelleRuntimeCPU::singlePrimaryVariable(const CollOfScalars& initial_values)
{
    std::vector<int> block_pattern;
    block_pattern.push_back(initial_values.size());
    // Syntax below is: CollOfScalarsAD::variable(block index, initialized from, block structure)
    return CollOfScalarsAD::variable(0, initial_values, block_pattern);
}