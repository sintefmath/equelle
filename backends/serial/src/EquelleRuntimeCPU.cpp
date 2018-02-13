/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "equelle/EquelleRuntimeCPU.hpp"
#include <opm/common/ErrorMacros.hpp>
#include <opm/grid/utility/StopWatch.hpp>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>




namespace equelle {

Opm::GridManager* createGridManager(const Opm::ParameterGroup& param)
{
    if (param.has("grid_filename")) {
        return new Opm::GridManager(param.get<std::string>("grid_filename"));
    }
    const int grid_dim = param.getDefault("grid_dim", 2);
    int num[3] = { 6, 1, 1 };
    double size[3] = { 1.0, 1.0, 1.0 };
    switch (grid_dim) { // Fall-throughs are intentional in this
    case 3:
        num[2] = param.getDefault("nz", num[2]);
        size[2] = param.getDefault("dz", size[2]);
    case 2:
        num[1] = param.getDefault("ny", num[1]);
        size[1] = param.getDefault("dy", size[1]);
        num[0] = param.getDefault("nx", num[0]);
        size[0] = param.getDefault("dx", size[0]);
        break;
    default:
        OPM_THROW(std::runtime_error, "Cannot handle " << grid_dim << " dimensions.");
    }
    switch (grid_dim) {
    case 2:
        return new Opm::GridManager(num[0], num[1], size[0], size[1]);
    case 3:
        return new Opm::GridManager(num[0], num[1], num[2], size[0], size[1], size[2]);
    default:
        OPM_THROW(std::runtime_error, "Cannot handle " << grid_dim << " dimensions.");
    }
}




EquelleRuntimeCPU::EquelleRuntimeCPU(const Opm::ParameterGroup& param)
    : grid_manager_(equelle::createGridManager(param)),
      grid_(*(grid_manager_->c_grid())),
      ops_(grid_),
      linsolver_(param),
      output_to_file_(param.getDefault("output_to_file", false)),
      verbose_(param.getDefault("verbose", 0)),
      param_(param),
      max_iter_(param.getDefault("max_iter", 10)),
      abs_res_tol_(param.getDefault("abs_res_tol", 1e-6))
{
}

EquelleRuntimeCPU::EquelleRuntimeCPU(const UnstructuredGrid *grid, const Opm::ParameterGroup &param)
    : grid_( *grid ),
      ops_(grid_),
      linsolver_(param),
      output_to_file_(param.getDefault("output_to_file", false)),
      verbose_(param.getDefault("verbose", 0)),
      param_(param),
      max_iter_(param.getDefault("max_iter", 10)),
      abs_res_tol_(param.getDefault("abs_res_tol", 1e-6))
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
        if (grid_.face_cells[2*face] == Boundary::outer || grid_.face_cells[2*face + 1] == Boundary::outer ) {
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
    CollOfScalar::V areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = grid_.face_areas[faces[i].index];
    }
    return areas;
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfCell& cells) const
{
    const int n = cells.size();
    CollOfScalar::V volumes(n);
    for (int i = 0; i < n; ++i) {
        volumes[i] = grid_.cell_volumes[cells[i].index];
    }
    return volumes;
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfVector& vectors) const
{
    CollOfScalar norm2 = vectors.col(0) * vectors.col(0);
    const int dim = vectors.numCols();
    for (int d = 1; d < dim; ++d) {
        norm2 += vectors.col(d) * vectors.col(d);
    }
    return sqrt(norm2);
}


CollOfScalar EquelleRuntimeCPU::norm(const CollOfScalar& scalars) const
{
    const CollOfScalar::V& v = scalars.value();
    const int sz = v.size();
    CollOfScalar::V factors(sz);
    for (int i = 0; i < sz; ++i) {
        if (v[i] >= 0.0) {
            factors[i] = 1.0;
        } else {
            factors[i] = -1.0;
        }
    }
    return factors * scalars;
}


CollOfVector EquelleRuntimeCPU::centroid(const CollOfFace& faces) const
{
    const int n = faces.size();
    const int dim = grid_.dimensions;
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> c(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.face_centroids + dim * faces[i].index;
        for (int d = 0; d < dim; ++d) {
            c(i, d) = fc[d];
        }
    }
    CollOfVector centroids(dim);
    for (int d = 0; d < dim; ++d) {
        centroids.col(d) = CollOfScalar(c.col(d));
    }
    return centroids;
}


CollOfVector EquelleRuntimeCPU::centroid(const CollOfCell& cells) const
{
    const int n = cells.size();
    const int dim = grid_.dimensions;
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> c(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fc = grid_.cell_centroids + dim * cells[i].index;
        for (int d = 0; d < dim; ++d) {
            c(i, d) = fc[d];
        }
    }
    CollOfVector centroids(dim);
    for (int d = 0; d < dim; ++d) {
        centroids.col(d) = CollOfScalar(c.col(d));
    }
    return centroids;
}


CollOfVector EquelleRuntimeCPU::normal(const CollOfFace& faces) const
{
    const int n = faces.size();
    const int dim = grid_.dimensions;
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> nor(n, dim);
    for (int i = 0; i < n; ++i) {
        const double* fn = grid_.face_normals + dim * faces[i].index;
        for (int d = 0; d < dim; ++d) {
            nor(i, d) = fn[d];
        }
    }
    // Since the UnstructuredGrid uses the unorthodox convention that face
    // normals are scaled with the face areas, we must renormalize them.
    nor.colwise() /= nor.matrix().rowwise().norm().array();
    CollOfVector normals(dim);
    for (int d = 0; d < dim; ++d) {
        normals.col(d) = CollOfScalar(nor.col(d));
    }
    return normals;
}


CollOfScalar EquelleRuntimeCPU::sqrt(const CollOfScalar& x) const
{
    return equelle::sqrt(x);
}

CollOfScalar EquelleRuntimeCPU::dot(const CollOfVector& v1, const CollOfVector& v2) const
{
    if (v1.numCols() != v2.numCols()) {
        OPM_THROW(std::logic_error, "Non-matching dimension of Vectors for dot().");
    }
    if (v1.col(0).size() != v2.col(0).size()) {
        OPM_THROW(std::logic_error, "Non-matching size of Vector collections for dot().");
    }
    const int dim = grid_.dimensions;
    CollOfScalar result = v1.col(0) * v2.col(0);
    for (int d = 1; d < dim; ++d) {
        result += v1.col(d) * v2.col(d);
    }
    return result;
}


CollOfScalar EquelleRuntimeCPU::gradient(const CollOfScalar& cell_scalarfield) const
{
    return ops_.grad * cell_scalarfield;//.matrix();
}


CollOfScalar EquelleRuntimeCPU::negGradient(const CollOfScalar& cell_scalarfield) const
{
    return ops_.ngrad * cell_scalarfield;//.matrix();
}


CollOfScalar EquelleRuntimeCPU::divergence(const CollOfScalar& face_fluxes) const
{
    if (face_fluxes.size() == ops_.internal_faces.size()) {
        // This is actually a hack, the compiler should know to emit interiorDivergence()
        // eventually, but as a temporary measure we do this.
        return interiorDivergence(face_fluxes);
    }
    return ops_.fulldiv * face_fluxes;//.matrix();
}


CollOfScalar EquelleRuntimeCPU::interiorDivergence(const CollOfScalar& face_fluxes) const
{
    return ops_.div * face_fluxes;//.matrix();
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

Scalar EquelleRuntimeCPU::minReduce(const CollOfScalar& x) const
{
    return x.value().minCoeff();
}

Scalar EquelleRuntimeCPU::maxReduce(const CollOfScalar& x) const
{
    return x.value().maxCoeff();
}

Scalar EquelleRuntimeCPU::sumReduce(const CollOfScalar& x) const
{
    return x.value().sum();
}

Scalar EquelleRuntimeCPU::prodReduce(const CollOfScalar& x) const
{
    return x.value().prod();
}

CollOfScalar EquelleRuntimeCPU::solveForUpdate(const CollOfScalar& residual) const
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr;
    residual.derivative()[0].toSparse(matr);

    CollOfScalar::V du = CollOfScalar::V::Zero(residual.size());

    Opm::time::StopWatch clock;
    clock.start();

    // solve(n, # nonzero values ("val"), ptr to col indices
    // ("col_ind"), ptr to row locations in val array ("row_ind")
    // (these two may be swapped, not sure about the naming convention
    // here...), array of actual values ("val") (I guess... '*sa'...),
    // rhs, solution)
    Opm::LinearSolverInterface::LinearSolverReport rep
            = linsolver_.solve(matr.rows(), matr.nonZeros(),
                               matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                               residual.value().data(), du.data());

    if (verbose_ > 2) {
        std::cout << "        solveForUpdate: Linear solver took: " << clock.secsSinceLast() << " seconds." << std::endl;
    }
    if (!rep.converged) {
        OPM_THROW(std::runtime_error, "Linear solver convergence failure.");
    }
    return du;
}


double EquelleRuntimeCPU::twoNorm(const CollOfScalar& vals) const
{
    return vals.value().matrix().norm();
}


void EquelleRuntimeCPU::output(const String& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
}


void EquelleRuntimeCPU::output(const String& tag, const CollOfScalar& vals)
{
    if (output_to_file_) {
        int count = -1;
        auto it = outputcount_.find(tag);
        if (it == outputcount_.end()) {
            count = 0;
            outputcount_[tag] = 1; // should contain the count to be used next time for same tag.
        } else {
            count = outputcount_[tag];
            ++outputcount_[tag];
        }
        std::ostringstream fname;
        fname << tag << "-" << std::setw(5) << std::setfill('0') << count << ".output";
        std::ofstream file(fname.str().c_str());
        if (!file) {
            OPM_THROW(std::runtime_error, "Failed to open " << fname.str());
        }
        file.precision(16);
        std::copy(vals.value().data(), vals.value().data() + vals.size(), std::ostream_iterator<double>(file, "\n"));
    } else {
        std::cout << tag << " =\n";
        for (int i = 0; i < vals.size(); ++i) {
            std::cout << std::setw(15) << std::left << ( vals.value()[i] ) << " ";
        }
        std::cout << std::endl;
    }
}


Scalar EquelleRuntimeCPU::inputScalarWithDefault(const String& name,
                                                 const Scalar default_value)
{
    return param_.getDefault(name, default_value);
}


CollOfFace EquelleRuntimeCPU::inputDomainSubsetOf(const String& name,
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


CollOfCell EquelleRuntimeCPU::inputDomainSubsetOf(const String& name,
                                                  const CollOfCell& cell_superset)
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfCell data;
    for (auto it = beg; it != end; ++it) {
        data.push_back(Cell(*it));
    }
    if (!is_sorted(data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Input set of cells was not sorted in ascending order.");
    }
    if (!includes(cell_superset.begin(), cell_superset.end(), data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Given cells are not in the assumed subset.");
    }
    return data;
}


SeqOfScalar EquelleRuntimeCPU::inputSequenceOfScalar(const String& name)
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



void EquelleRuntimeCPU::ensureGridDimensionMin(const int minimum_grid_dimension) const
{
    if (grid_.dimensions < minimum_grid_dimension) {
        OPM_THROW(std::runtime_error, "Equelle simulator requires minimum " << minimum_grid_dimension
                  << " dimensions, but grid only has " << grid_.dimensions << " dimensions.");
    }
}



CollOfScalar EquelleRuntimeCPU::singlePrimaryVariable(const CollOfScalar& initial_values)
{
    std::vector<int> block_pattern;
    block_pattern.push_back(initial_values.size());
    // Syntax below is: CollOfScalar::variable(block index, initialized from, block structure)
    return CollOfScalar::variable(0, initial_values.value(), block_pattern);
}

} // equelle-namespace
