/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeCPU.hpp"
#include <opm/core/utility/ErrorMacros.hpp>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>


EquelleRuntimeCPU::EquelleRuntimeCPU(const Opm::parameter::ParameterGroup& param)
    : grid_from_file_(param.has("grid_filename")),
      grid_manager_(grid_from_file_ ? new Opm::GridManager(param.get<std::string>("grid_filename"))
                    : new Opm::GridManager(param.getDefault("n", 6), 1)),
      grid_(*(grid_manager_->c_grid())),
      ops_(grid_),
      linsolver_(param),
      output_to_file_(param.getDefault("output_to_file", false))
{
}


CollOfCells EquelleRuntimeCPU::allCells() const
{
    const int nc = grid_.number_of_cells;
    CollOfCells cells(nc);
    for (int c = 0; c < nc; ++c) {
        cells[c].index = c;
    }
    return cells;
}


// Note that this will not produce the expected results for a 1D grid realized as a 2D grid of dimension (n, 1) or (1, n), since all cells
// of such a grid are boundary cells.
//
// A hack could be made by testing for "three boundary faces" instead of "at least one boundary face" in the test inside the inner loop.

bool EquelleRuntimeCPU::boundaryCell(const int cell_index) const
{
    const int fp = grid_.cell_facepos[cell_index];
    bool boundary_cell = false;
    const int f_per_c = grid_.dimensions*2; // Can we assume this, or should we use cell_facepos[c+1]-cell_facepos[c]?!
#if 1
    for (int i=0; (i<f_per_c) && (!boundary_cell); ++i) {
        const int f = grid_.cell_faces[ fp+i ];
        boundary_cell = (grid_.face_cells[2*f]==-1) || (grid_.face_cells[2*f + 1]==-1);
    }
#else
    // Hack for 1D grid realized as 2D (n, 1) or (1, n)
    int num_of_boundary_faces = 0;
    for (int i=0; i<f_per_c; ++i) {
        const int f = grid_.cell_faces[ fp+i ];
        if ( (grid_.face_cells[2*f]==-1) || (grid_.face_cells[2*f + 1]==-1) ) {
            ++num_of_boundary_faces;
        }
    }
    if (num_of_boundary_faces>=3) {
        boundary_cell = true;
    }
#endif
    return boundary_cell;
}


CollOfCells EquelleRuntimeCPU::boundaryCells() const
{
    CollOfCells cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfCells EquelleRuntimeCPU::interiorCells() const
{
    CollOfCells cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( !boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfFaces EquelleRuntimeCPU::allFaces() const
{
    const int nf = grid_.number_of_faces;
    CollOfFaces faces(nf);
    for (int f = 0; f < nf; ++f) {
        faces[f].index = f;
    }
    return faces;
}


// Again... this is kind of botched for a 1D grid implemented as a 2D(n, 1) or 2D(1, n) grid...

CollOfFaces EquelleRuntimeCPU::boundaryFaces() const
{
    const int nif = ops_.internal_faces.size();
    const int nbf = grid_.number_of_faces - nif;
    CollOfFaces bfaces(nbf);
    int if_cursor = 0;
    int bf_cursor = 0;

    // This works as long as ops_.internal_faces(i)<ops_.internal_faces(i+1), which it currently is.
    // Would be better to extend HelperOps to support this functionality.

    for (int i = 0; i < grid_.number_of_faces; ++i) {
        // Advance if_cursor so that the next internal face to look out for has larger or equal index to i
        while ( (if_cursor < nif) && (i > ops_.internal_faces[if_cursor]) ) {
            ++if_cursor;
        }
        // Now if_cursor points beyond the last internal face, or internal_face[if_cursor]>=i.
        // If (if_cursor points beyond the last internal face) or (internal_face[if_cursor] is truly > i), we surely have a boundary face...
        if ( (if_cursor == nif) || (ops_.internal_faces[if_cursor] > i) ) {
            bfaces[bf_cursor].index = i;
            ++bf_cursor;
        }
    }

    return bfaces;
}


CollOfFaces EquelleRuntimeCPU::interiorFaces() const
{
    const int nif = ops_.internal_faces.size();
    CollOfFaces ifaces(nif);
    for (int i = 0; i < nif; ++i) {
        ifaces[i].index = ops_.internal_faces(i);
    }
    return ifaces;
}


CollOfCells EquelleRuntimeCPU::firstCell(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfCells fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index];
    }
    return fcells;
}


CollOfCells EquelleRuntimeCPU::secondCell(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfCells fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index + 1];
    }
    return fcells;
}


CollOfScalars EquelleRuntimeCPU::norm(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfScalars areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = grid_.face_areas[faces[i].index];
    }
    return areas;
}


CollOfScalars EquelleRuntimeCPU::norm(const CollOfCells& cells) const
{
    const int n = cells.size();
    CollOfScalars volumes(n);
    for (int i = 0; i < n; ++i) {
        volumes[i] = grid_.cell_volumes[cells[i].index];
    }
    return volumes;
}


CollOfScalars EquelleRuntimeCPU::norm(const CollOfVectors& vectors) const
{
    return vectors.matrix().rowwise().norm();
}


CollOfVectors EquelleRuntimeCPU::centroid(const CollOfFaces& faces) const
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


CollOfVectors EquelleRuntimeCPU::centroid(const CollOfCells& cells) const
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


CollOfScalars EquelleRuntimeCPU::gradient(const CollOfScalars& cell_scalarfield) const
{
    return ops_.grad * cell_scalarfield.matrix();
}


CollOfScalarsAD EquelleRuntimeCPU::gradient(const CollOfScalarsAD& cell_scalarfield) const
{
    return ops_.grad * cell_scalarfield;
}


CollOfScalars EquelleRuntimeCPU::negGradient(const CollOfScalars& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield.matrix();
}


CollOfScalarsAD EquelleRuntimeCPU::negGradient(const CollOfScalarsAD& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield;
}


CollOfScalars EquelleRuntimeCPU::divergence(const CollOfScalars& face_fluxes) const
{
    if (face_fluxes.size() == ops_.internal_faces.size()) {
        // This is actually a hack, the compiler should know to emit interiorDivergence()
        // eventually, but as a temporary measure we do this.
        return interiorDivergence(face_fluxes);
    }
    return ops_.fulldiv * face_fluxes.matrix();
}


CollOfScalarsAD EquelleRuntimeCPU::divergence(const CollOfScalarsAD& face_fluxes) const
{
    if (face_fluxes.size() == ops_.internal_faces.size()) {
        // This is actually a hack, the compiler should know to emit interiorDivergence()
        // eventually, but as a temporary measure we do this.
        return interiorDivergence(face_fluxes);
    }
    return ops_.fulldiv * face_fluxes;
}


CollOfScalars EquelleRuntimeCPU::interiorDivergence(const CollOfScalars& face_fluxes) const
{
    return ops_.div * face_fluxes.matrix();
}


CollOfScalarsAD EquelleRuntimeCPU::interiorDivergence(const CollOfScalarsAD& face_fluxes) const
{
    return ops_.div * face_fluxes;
}


CollOfBooleans EquelleRuntimeCPU::isEmpty(const CollOfCells& cells) const
{
    const size_t sz = cells.size();
    CollOfBooleans retval = CollOfBooleans::Constant(sz, false);
    for (size_t i = 0; i < sz; ++i) {
        if (cells[i].index < 0) {
            retval[i] = true;
        }
    }
    return retval;
}


CollOfBooleans EquelleRuntimeCPU::isEmpty(const CollOfFaces& faces) const
{
    const size_t sz = faces.size();
    CollOfBooleans retval = CollOfBooleans::Constant(sz, false);
    for (size_t i = 0; i < sz; ++i) {
        if (faces[i].index < 0) {
            retval[i] = true;
        }
    }
    return retval;
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


double EquelleRuntimeCPU::twoNorm(const CollOfScalars& vals) const
{
    return vals.matrix().norm();
}


double EquelleRuntimeCPU::twoNorm(const CollOfScalarsAD& vals) const
{
    return twoNorm(vals.value());
}


void EquelleRuntimeCPU::output(const std::string& tag, const double val) const
{
    std::cout << tag << val << std::endl;
}


void EquelleRuntimeCPU::output(const std::string& tag, const CollOfScalars& vals) const
{
    if (output_to_file_) {
        std::string filename = tag + ".output";
        std::ofstream os(filename.c_str());
        for (int i = 0; i < vals.size(); ++i) {
            os << std::setw(15) << std::left << ( vals[i] ) << " ";
        }
        os << std::endl;
    } else {
        std::cout << tag;
        for (int i = 0; i < vals.size(); ++i) {
            std::cout << std::setw(15) << std::left << ( vals[i] ) << " ";
        }
        std::cout << std::endl;
    }
}


void EquelleRuntimeCPU::output(const std::string& tag, const CollOfScalarsOnColl& vals) const
{
    output(tag, vals.getColl());
    std::cout << "(This was On Collection " << vals.getOnColl() << ")" << std::endl;
}


void EquelleRuntimeCPU::output(const std::string& tag, const CollOfScalarsAD& vals) const
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


CollOfFaces EquelleRuntimeCPU::getUserSpecifiedCollectionOfFaceSubsetOf(const Opm::parameter::ParameterGroup& param,
                                                                        const std::string& name,
                                                                        const CollOfFaces& face_superset)
{
    const std::string filename = param.get<std::string>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        THROW("Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfFaces data;
    for (auto it = beg; it != end; ++it) {
        data.push_back(Face(*it));
    }
    if (!is_sorted(data.begin(), data.end())) {
        THROW("Input set of faces was not sorted in ascending order.");
    }
    if (!includes(face_superset.begin(), face_superset.end(), data.begin(), data.end())) {
        THROW("Given faces are not in the assumed subset.");
    }
    return data;
}


CollOfScalarsAD EquelleRuntimeCPU::singlePrimaryVariable(const CollOfScalars& initial_values)
{
    std::vector<int> block_pattern;
    block_pattern.push_back(initial_values.size());
    // Syntax below is: CollOfScalarsAD::variable(block index, initialized from, block structure)
    return CollOfScalarsAD::variable(0, initial_values, block_pattern);
}
