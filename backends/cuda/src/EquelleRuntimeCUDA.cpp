/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeCUDA.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"
#include "CollOfScalar.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "CollOfVector.hpp"
#include "wrapEquelleRuntime.hpp"

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
      ops_(grid_),
      devOps_(ops_.grad, ops_.div, ops_.fulldiv,
	      ops_.internal_faces.rows()),
      linsolver_(param),
      output_to_file_(param.getDefault("output_to_file", false)),
      verbose_(param.getDefault("verbose", 0)),
      param_(param),
      max_iter_(param.getDefault("max_iter", 10)),
      abs_res_tol_(param.getDefault("abs_res_tol", 1e-6))
{
    wrapEquelleRuntimeCUDA::init_cusparse();
}

// Destructor:
EquelleRuntimeCUDA::~EquelleRuntimeCUDA() {
    wrapEquelleRuntimeCUDA::destroy_cusparse();
}


CollOfCell EquelleRuntimeCUDA::allCells() const
{
    return dev_grid_.allCells();
}

CollOfCell EquelleRuntimeCUDA::boundaryCells() const 
{
    return dev_grid_.boundaryCells();
}

CollOfCell EquelleRuntimeCUDA::interiorCells() const 
{
    return dev_grid_.interiorCells();
}

CollOfFace EquelleRuntimeCUDA::allFaces() const
{
    return dev_grid_.allFaces();
}

CollOfFace EquelleRuntimeCUDA::boundaryFaces() const
{
    return dev_grid_.boundaryFaces();
}

CollOfFace EquelleRuntimeCUDA::interiorFaces() const
{
    return dev_grid_.interiorFaces();
}

CollOfCell EquelleRuntimeCUDA::firstCell(CollOfFace faces) const
{
    return dev_grid_.firstCell(faces);
}

CollOfCell EquelleRuntimeCUDA::secondCell(CollOfFace faces) const
{
    return dev_grid_.secondCell(faces);
}


CollOfScalar EquelleRuntimeCUDA::norm(const CollOfVector& vectors) const
{
    return vectors.norm();
}

CollOfVector EquelleRuntimeCUDA::normal(const CollOfFace& faces) const
{
    return dev_grid_.normal(faces);
}

CollOfScalar EquelleRuntimeCUDA::dot( const CollOfVector& v1,
				      const CollOfVector& v2 ) const 
{
    return v1.dot(v2);
}




void EquelleRuntimeCUDA::output(const String& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
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


