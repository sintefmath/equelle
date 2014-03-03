/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"
#include "CollOfScalar.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"


#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/utility/StopWatch.hpp>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>

using namespace equelleCUDA;


namespace
{
    Opm::GridManager* createGridManager(const Opm::parameter::ParameterGroup& param)
    {
        if (param.has("grid_filename")) {
        	// Unstructured grid
            return new Opm::GridManager(param.get<std::string>("grid_filename"));
        }
        // Otherwise: Cartesian grid
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
} // anonymous namespace



EquelleRuntimeCUDA::EquelleRuntimeCUDA(const Opm::parameter::ParameterGroup& param)
    : grid_manager_(createGridManager(param)),
      grid_(*(grid_manager_->c_grid())),
      dev_grid_(UnstructuredGrid(*(grid_manager_->c_grid()))),
      //ops_(grid_),
      linsolver_(param),
      output_to_file_(param.getDefault("output_to_file", false)),
      verbose_(param.getDefault("verbose", 0)),
      param_(param),
      max_iter_(param.getDefault("max_iter", 10)),
      abs_res_tol_(param.getDefault("abs_res_tol", 1e-6))
{
}


//CollOfCell EquelleRuntimeCUDA::allCells() const
//{
//    return dev_grid_.allCells();
    //return boundaryCells();
//}


// Note that this will not produce what some would consider the expected results for a 1D grid realized as a 2D grid of dimension (n, 1) or (1, n), since all cells
// of such a grid are boundary cells.
// That points out that communicating the grid concepts is very important.
bool EquelleRuntimeCUDA::boundaryCell(const int cell_index) const
{
    for (int hface = grid_.cell_facepos[cell_index]; hface < grid_.cell_facepos[cell_index + 1]; ++hface) {
        const int face = grid_.cell_faces[hface];
        if (grid_.face_cells[2*face] < 0 || grid_.face_cells[2*face + 1] < 0) {
            return true;
        }
    }
    return false;
}


CollOfCellCPU EquelleRuntimeCUDA::boundaryCells() const
{
    CollOfCellCPU cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfCellCPU EquelleRuntimeCUDA::interiorCells() const
{
    CollOfCellCPU cells;
    const int nc = grid_.number_of_cells;
    cells.reserve(nc);
    for (int c = 0; c < nc; ++c) {
        if ( !boundaryCell(c) ) {
            cells.emplace_back( Cell(c) );
        }
    }
    return cells;
}


CollOfFaceCPU EquelleRuntimeCUDA::allFaces() const
{
    const int nf = grid_.number_of_faces;
    CollOfFaceCPU faces(nf);
    for (int f = 0; f < nf; ++f) {
        faces[f].index = f;
    }
    return faces;
}


// Again... this is kind of botched for a 1D grid implemented as a 2D(n, 1) or 2D(1, n) grid...





CollOfCellCPU EquelleRuntimeCUDA::firstCell(const CollOfFaceCPU& faces) const
{
    const int n = faces.size();
    CollOfCellCPU fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index];
    }
    return fcells;
}


CollOfCellCPU EquelleRuntimeCUDA::secondCell(const CollOfFaceCPU& faces) const
{
    const int n = faces.size();
    CollOfCellCPU fcells(n);
    for (int i = 0; i < n; ++i) {
        fcells[i].index = grid_.face_cells[2*faces[i].index + 1];
    }
    return fcells;
}





CollOfVector EquelleRuntimeCUDA::centroid(const CollOfFaceCPU& faces) const
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
        //centroids.col(d) = CollOfScalarCPU(c.col(d));
    }
    return centroids;
}


CollOfVector EquelleRuntimeCUDA::centroid(const CollOfCellCPU& cells) const
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
        //centroids.col(d) = CollOfScalarCPU(c.col(d));
    }
    return centroids;
}


CollOfVector EquelleRuntimeCUDA::normal(const CollOfFaceCPU& faces) const
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
        //normals.col(d) = CollOfScalarCPU(nor.col(d));
    }
    return normals;
}




CollOfScalarCPU EquelleRuntimeCUDA::dot(const CollOfVector& v1, const CollOfVector& v2) const
{
    if (v1.numCols() != v2.numCols()) {
        OPM_THROW(std::logic_error, "Non-matching dimension of Vectors for dot().");
    }
    if (v1.col(0).size() != v2.col(0).size()) {
        OPM_THROW(std::logic_error, "Non-matching size of Vector collections for dot().");
    }
    const int dim = grid_.dimensions;
    CollOfScalarCPU result = v1.col(0);// * v2.col(0);
    for (int d = 1; d < dim; ++d) {
	// result += v1.col(d) * v2.col(d);
    }
    return result;
}




void EquelleRuntimeCUDA::output(const String& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
}


Scalar EquelleRuntimeCUDA::inputScalarWithDefault(const String& name,
                                                         const Scalar default_value)
{
    return param_.getDefault(name, default_value);
}


CollOfFaceCPU EquelleRuntimeCUDA::inputDomainSubsetOf(const String& name,
                                                  const CollOfFaceCPU& face_superset)
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfFaceCPU data;
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


CollOfCellCPU EquelleRuntimeCUDA::inputDomainSubsetOf(const String& name,
                                                  const CollOfCellCPU& cell_superset)
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfCellCPU data;
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


SeqOfScalar EquelleRuntimeCUDA::inputSequenceOfScalar(const String& name)
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



void EquelleRuntimeCUDA::ensureGridDimensionMin(const int minimum_grid_dimension) const
{
    if (grid_.dimensions < minimum_grid_dimension) {
        OPM_THROW(std::runtime_error, "Equelle simulator requires minimum " << minimum_grid_dimension
                  << " dimensions, but grid only has " << grid_.dimensions << " dimensions.");
    }
}



// HAVAHOL: added function for doing testing
UnstructuredGrid EquelleRuntimeCUDA::getGrid() const {
	return grid_;
}


