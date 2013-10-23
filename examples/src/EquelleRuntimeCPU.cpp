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
      output_to_file_(param.getDefault("output_to_file", false)),
      verbose_(param.getDefault("verbose", 0)),
      param_(param)
{
}


CollOfCell EquelleRuntimeCPU::allCells() const
{
    const int nc = grid_.number_of_cells;
    CollOfCell cells(nc);
    for (int c = 0; c < nc; ++c) {
        cells[c].index = c;
    }
    return cells;
}


// Note that this will not produce what some would consider the expected results for a 1D grid realized as a 2D grid of dimension (n, 1) or (1, n), since all cells
// of such a grid are boundary cells.
// That points out that communicating the grid concepts is very important.
bool EquelleRuntimeCPU::boundaryCell(const int cell_index) const
{
    for (int hface = grid_.cell_facepos[cell_index]; hface < grid_.cell_facepos[cell_index + 1]; ++hface) {
        const int face = grid_.cell_faces[hface];
        if (grid_.face_cells[2*face] < 0 || grid_.face_cells[2*face + 1] < 0) {
            return true;
        }
    }
    return false;
}


CollOfCell EquelleRuntimeCPU::boundaryCells() const
{
    CollOfCell cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfCell EquelleRuntimeCPU::interiorCells() const
{
    CollOfCell cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( !boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfFace EquelleRuntimeCPU::allFaces() const
{
    const int nf = grid_.number_of_faces;
    CollOfFace faces(nf);
    for (int f = 0; f < nf; ++f) {
        faces[f].index = f;
    }
    return faces;
}


// Again... this is kind of botched for a 1D grid implemented as a 2D(n, 1) or 2D(1, n) grid...

CollOfFace EquelleRuntimeCPU::boundaryFaces() const
{
    const int nif = ops_.internal_faces.size();
    const int nbf = grid_.number_of_faces - nif;
    CollOfFace bfaces(nbf);
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


CollOfFace EquelleRuntimeCPU::interiorFaces() const
{
    const int nif = ops_.internal_faces.size();
    CollOfFace ifaces(nif);
    for (int i = 0; i < nif; ++i) {
        ifaces[i].index = ops_.internal_faces(i);
    }
    return ifaces;
}


CollOfCell EquelleRuntimeCPU::firstCell(const CollOfFace& faces) const
{
    const int n = faces.size();
    CollOfCell fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index];
    }
    return fcells;
}


CollOfCell EquelleRuntimeCPU::secondCell(const CollOfFace& faces) const
{
    const int n = faces.size();
    CollOfCell fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index + 1];
    }
    return fcells;
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfFace& faces) const
{
    const int n = faces.size();
    CollOfScalar areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = grid_.face_areas[faces[i].index];
    }
    return areas;
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfCell& cells) const
{
    const int n = cells.size();
    CollOfScalar volumes(n);
    for (int i = 0; i < n; ++i) {
        volumes[i] = grid_.cell_volumes[cells[i].index];
    }
    return volumes;
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfVector& vectors) const
{
    return vectors.matrix().rowwise().norm();
}


CollOfVector EquelleRuntimeCPU::centroid(const CollOfFace& faces) const
{
    const int n = faces.size();
    const int dim = grid_.dimensions;
    CollOfVector centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.face_centroids + dim * faces[i].index;
        for (int d = 0; d < dim; ++d) {
            centroids(i, d) = fc[d];
        }
    }
    return centroids;
}


CollOfVector EquelleRuntimeCPU::centroid(const CollOfCell& cells) const
{
    const int n = cells.size();
    const int dim = grid_.dimensions;
    CollOfVector centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.cell_centroids + dim * cells[i].index;
        for (int d = 0; d < dim; ++d) {
            centroids(i, d) = fc[d];
        }
    }
    return centroids;
}


CollOfScalar EquelleRuntimeCPU::gradient(const CollOfScalar& cell_scalarfield) const
{
    return ops_.grad * cell_scalarfield.matrix();
}


CollOfScalarAD EquelleRuntimeCPU::gradient(const CollOfScalarAD& cell_scalarfield) const
{
    return ops_.grad * cell_scalarfield;
}


CollOfScalar EquelleRuntimeCPU::negGradient(const CollOfScalar& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield.matrix();
}


CollOfScalarAD EquelleRuntimeCPU::negGradient(const CollOfScalarAD& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield;
}


CollOfScalar EquelleRuntimeCPU::divergence(const CollOfScalar& face_fluxes) const
{
    if (face_fluxes.size() == ops_.internal_faces.size()) {
        // This is actually a hack, the compiler should know to emit interiorDivergence()
        // eventually, but as a temporary measure we do this.
        return interiorDivergence(face_fluxes);
    }
    return ops_.fulldiv * face_fluxes.matrix();
}


CollOfScalarAD EquelleRuntimeCPU::divergence(const CollOfScalarAD& face_fluxes) const
{
    if (face_fluxes.size() == ops_.internal_faces.size()) {
        // This is actually a hack, the compiler should know to emit interiorDivergence()
        // eventually, but as a temporary measure we do this.
        return interiorDivergence(face_fluxes);
    }
    return ops_.fulldiv * face_fluxes;
}


CollOfScalar EquelleRuntimeCPU::interiorDivergence(const CollOfScalar& face_fluxes) const
{
    return ops_.div * face_fluxes.matrix();
}


CollOfScalarAD EquelleRuntimeCPU::interiorDivergence(const CollOfScalarAD& face_fluxes) const
{
    return ops_.div * face_fluxes;
}


CollOfBool EquelleRuntimeCPU::isEmpty(const CollOfCell& cells) const
{
    const size_t sz = cells.size();
    CollOfBool retval = CollOfBool::Constant(sz, false);
    for (size_t i = 0; i < sz; ++i) {
        if (cells[i].index < 0) {
            retval[i] = true;
        }
    }
    return retval;
}


CollOfBool EquelleRuntimeCPU::isEmpty(const CollOfFace& faces) const
{
    const size_t sz = faces.size();
    CollOfBool retval = CollOfBool::Constant(sz, false);
    for (size_t i = 0; i < sz; ++i) {
        if (faces[i].index < 0) {
            retval[i] = true;
        }
    }
    return retval;
}


CollOfScalar EquelleRuntimeCPU::solveForUpdate(const CollOfScalarAD& residual) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr = residual.derivative()[0];

    CollOfScalar du = CollOfScalar::Zero(residual.size());

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
        OPM_THROW(std::runtime_error, "Linear solver convergence failure.");
    }
    return du;
}


double EquelleRuntimeCPU::twoNorm(const CollOfScalar& vals) const
{
    return vals.matrix().norm();
}


double EquelleRuntimeCPU::twoNorm(const CollOfScalarAD& vals) const
{
    return twoNorm(vals.value());
}


void EquelleRuntimeCPU::output(const String& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
}


void EquelleRuntimeCPU::output(const String& tag, const CollOfScalar& vals) const
{
    if (output_to_file_) {
        String filename = tag + ".output";
        std::ofstream os(filename.c_str());
        for (int i = 0; i < vals.size(); ++i) {
            os << std::setw(15) << std::left << ( vals[i] ) << " ";
        }
        os << std::endl;
    } else {
        std::cout << tag << " =\n";
        for (int i = 0; i < vals.size(); ++i) {
            std::cout << std::setw(15) << std::left << ( vals[i] ) << " ";
        }
        std::cout << std::endl;
    }
}


void EquelleRuntimeCPU::output(const String& tag, const CollOfScalarAD& vals) const
{
    output(tag, vals.value());
}


Scalar EquelleRuntimeCPU::userSpecifiedScalarWithDefault(const String& name,
                                                         const Scalar default_value)
{
    return param_.getDefault(name, default_value);
}


CollOfFace EquelleRuntimeCPU::userSpecifiedCollectionOfFaceSubsetOf(const String& name,
                                                                     const CollOfFace& face_superset)
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfFace data;
    for (auto it = beg; it != end; ++it) {
        data.push_back(Face(*it));
    }
    if (!is_sorted(data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Input set of faces was not sorted in ascending order.");
    }
    if (!includes(face_superset.begin(), face_superset.end(), data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Given faces are not in the assumed subset.");
    }
    return data;
}


SeqOfScalar EquelleRuntimeCPU::userSpecifiedSequenceOfScalar(const String& name)
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<Scalar> beg(is);
    std::istream_iterator<Scalar> end;
    SeqOfScalar data(beg, end);
    return data;
}


CollOfScalarAD EquelleRuntimeCPU::singlePrimaryVariable(const CollOfScalar& initial_values)
{
    std::vector<int> block_pattern;
    block_pattern.push_back(initial_values.size());
    // Syntax below is: CollOfScalarAD::variable(block index, initialized from, block structure)
    return CollOfScalarAD::variable(0, initial_values, block_pattern);
}
