/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeDune.hpp"
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>


EquelleRuntimeDune::EquelleRuntimeDune(const Opm::parameter::ParameterGroup& param)
    : output_to_file_(param.getDefault("output_to_file", false))
{
    grid_.init(param);
    setupFaceSets();
}


CollOfCells EquelleRuntimeDune::allCells() const
{
    const int nc = grid_.size(0);
    CollOfCells cells;
    cells.reserve(nc);
    Dune::CpGrid::LeafGridView gv = grid_.leafView();
    for (auto cit = gv.begin<0>(); cit != gv.end<0>(); ++cit) {
        cells.push_back(cit->seed());
    }
    return cells;
}



CollOfFaces EquelleRuntimeDune::allFaces() const
{
    return allfaces_;
}


CollOfFaces EquelleRuntimeDune::boundaryFaces() const
{
    return bfaces_;
}


CollOfFaces EquelleRuntimeDune::interiorFaces() const
{
    return ifaces_;
}


CollOfCells EquelleRuntimeDune::firstCell(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfCells fcells;
    fcells.reserve(n);
    for (int i = 0; i < n; ++i) {
        fcells.emplace_back(faces[i].first_cell);
    }
    return fcells;
}


CollOfCells EquelleRuntimeDune::secondCell(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfCells fcells;
    fcells.reserve(n);
    for (int i = 0; i < n; ++i) {
        fcells.emplace_back(faces[i].second_cell);
    }
    return fcells;
}


CollOfScalars EquelleRuntimeDune::norm(const CollOfFaces& faces) const
{
    const int n = faces.size();
    CollOfScalars areas(n);
    for (int i = 0; i < n; ++i) {
        Cell c = faces[i].first_cell;
        int icount = 0;
        for (auto iit = c->ileafbegin(); iit != c->ileafend(); ++iit, ++icount) {
            if (icount == faces[i].intersection_number) {
                areas[i] = iit->geometry().volume();
            }
        }
    }
    return areas;
}


CollOfScalars EquelleRuntimeDune::norm(const CollOfCells& cells) const
{
    const int n = cells.size();
    CollOfScalars volumes(n);
    for (int i = 0; i < n; ++i) {
        volumes[i] = cells[i]->geometry().volume();
    }
    return volumes;
}


CollOfScalars EquelleRuntimeDune::norm(const CollOfVectors& vectors) const
{
    return vectors.matrix().rowwise().norm();
}


CollOfVectors EquelleRuntimeDune::centroid(const CollOfFaces& faces) const
{
    const int n = faces.size();
    const int dim = Dune::CpGrid::dimension;
    CollOfVectors centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        Cell c = faces[i].first_cell;
        int icount = 0;
        for (auto iit = c->ileafbegin(); iit != c->ileafend(); ++iit, ++icount) {
            if (icount == faces[i].intersection_number) {
                auto fc = iit->geometry().center();
                for (int d = 0; d < dim; ++d) {
                    centroids(i, d) = fc[d];
                }
            }
        }
    }
    return centroids;
}


CollOfVectors EquelleRuntimeDune::centroid(const CollOfCells& cells) const
{
    const int n = cells.size();
    const int dim = Dune::CpGrid::dimension;
    CollOfVectors centroids(n, dim);
    for (int i = 0; i < n; ++i) {
        auto cc = cells[i]->geometry().center();
        for (int d = 0; d < dim; ++d) {
            centroids(i, d) = cc[d];
        }
    }
    return centroids;
}


CollOfScalars EquelleRuntimeDune::gradient(const CollOfScalars& cell_scalarfield) const
{
    const int nf = ifaces_.size();
    CollOfScalars gr(nf);
    Dune::CpGrid::LeafGridView gv = grid_.leafView();
    auto iset = gv.indexSet();
    for (int f = 0; f < nf; ++f) {
        gr[f] = cell_scalarfield[iset.index(*(ifaces_[f].second_cell))]
            - cell_scalarfield[iset.index(*(ifaces_[f].first_cell))];
    }
    return gr;
}



CollOfScalars EquelleRuntimeDune::negGradient(const CollOfScalars& cell_scalarfield) const
{
    return -gradient(cell_scalarfield);
}


CollOfScalars EquelleRuntimeDune::divergence(const CollOfScalars& face_fluxes) const
{
    const CollOfFaces& fset = (size_t(face_fluxes.size()) == ifaces_.size()) ? ifaces_ : allfaces_;
    CollOfScalars div = CollOfScalars::Zero(grid_.size(0));
    const size_t nf = fset.size();
    Dune::CpGrid::LeafGridView gv = grid_.leafView();
    auto iset = gv.indexSet();
    // Warning: not parallelizable for loop.
    for (size_t f = 0; f < nf; ++f) {
        int c1 = iset.index(fset[f].first_cell);
        int c2 = iset.index(fset[f].second_cell);
        div[c1] += face_fluxes[f];
        div[c2] -= face_fluxes[f];
    }
    return div;
}


CollOfScalars EquelleRuntimeDune::interiorDivergence(const CollOfScalars& face_fluxes) const
{
    // Ok since that method covers both interior-only and all-faces cases.
    return divergence(face_fluxes);
}


CollOfBooleans EquelleRuntimeDune::isEmpty(const CollOfCells& cells) const
{
    const size_t sz = cells.size();
    const Cell empty = emptyCell();
    CollOfBooleans retval = CollOfBooleans::Constant(sz, false);
    for (size_t i = 0; i < sz; ++i) {
        if (cells[i] == empty) {
            retval[i] = true;
        }
    }
    return retval;
}


CollOfBooleans EquelleRuntimeDune::isEmpty(const CollOfFaces& faces) const
{
    OPM_THROW(std::runtime_error, "isEmpty() for CollOfFaces not implemented.");
}


double EquelleRuntimeDune::twoNorm(const CollOfScalars& vals) const
{
    return vals.matrix().norm();
}


void EquelleRuntimeDune::output(const std::string& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
}


void EquelleRuntimeDune::output(const std::string& tag, const CollOfScalars& vals) const
{
    if (output_to_file_) {
        std::string filename = tag + ".output";
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


CollOfScalars EquelleRuntimeDune::getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
                                                                    const std::string& name,
                                                                    const int size)
{
    const bool from_file = param.getDefault(name + "_from_file", false);
    if (from_file) {
        const std::string filename = param.get<std::string>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            OPM_THROW(std::runtime_error, "Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;
        std::vector<double> data(beg, end);
        if (int(data.size()) != size) {
            OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
        }
        return CollOfScalars(Eigen::Map<CollOfScalars>(&data[0], size));
    } else {
        // Uniform values.
        return CollOfScalars::Constant(size, param.get<double>(name));
    }
}


CollOfFaces EquelleRuntimeDune::getUserSpecifiedCollectionOfFaceSubsetOf(const Opm::parameter::ParameterGroup& param,
                                                                         const std::string& name,
                                                                         const CollOfFaces& face_superset)
{
    const std::string filename = param.get<std::string>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
        OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    CollOfFaces data;
    for (auto it = beg; it != end; ++it) {
        data.push_back(allfaces_[*it]);
    }
    if (!is_sorted(data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Input set of faces was not sorted in ascending order.");
    }
    if (!includes(face_superset.begin(), face_superset.end(), data.begin(), data.end())) {
        OPM_THROW(std::runtime_error, "Given faces are not in the assumed subset.");
    }
    return data;
}


Cell EquelleRuntimeDune::emptyCell() const
{
    // Use non-existing cell with index -1 for the outside.
    return Cell(grid_, -1, true);
}


void EquelleRuntimeDune::setupFaceSets()
{
    const int nf = grid_.size(1);
    allfaces_.clear();
    allfaces_.reserve(nf);
    ifaces_.clear();
    ifaces_.reserve(nf); // far too big, but we do not care for the moment
    bfaces_.clear();
    bfaces_.reserve(nf); // also too big
    Dune::CpGrid::LeafGridView gv = grid_.leafView();
    auto iset = gv.indexSet();
    for (auto cit = gv.begin<0>(); cit != gv.end<0>(); ++cit) {
        int icount = 0;
        for (auto iit = cit->ileafbegin(); iit != cit->ileafend(); ++iit, ++icount) {
            if (iit->neighbor()) {
                if (iset.index(iit->inside()) < iset.index(iit->outside())) {
                    allfaces_.emplace_back(cit->seed(), iit->outside().seed(), icount);
                    ifaces_.emplace_back(cit->seed(), iit->outside().seed(), icount);
                }
            } else {
                allfaces_.emplace_back(cit->seed(), emptyCell().seed(), icount);
                bfaces_.emplace_back(cit->seed(), emptyCell().seed(), icount);
            }
        }
    }
}
