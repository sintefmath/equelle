/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "EquelleRuntimeCUDA.hpp"
#include "CollOfScalar.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "CollOfVector.hpp"
#include "wrapEquelleRuntime.hpp"
#include "LinearSolver.hpp"

#include <opm/common/ErrorMacros.hpp>
#include <opm/core/utility/StopWatch.hpp>
#include <iomanip>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <set>
#include <tuple>

using namespace equelleCUDA;
using namespace wrapEquelleRuntimeCUDA;


namespace
{
    Opm::GridManager* createGridManager(const Opm::ParameterGroup& param)
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



EquelleRuntimeCUDA::EquelleRuntimeCUDA(const Opm::ParameterGroup& param)
    : grid_manager_(createGridManager(param)),
      grid_(*(grid_manager_->c_grid())),
      dev_grid_(grid_),
      devOps_(grid_),
      solver_(param.getDefault<std::string>("solver", "BiCGStab"),
              param.getDefault<std::string>("preconditioner", "diagonal"),
              param.getDefault("solver_max_iter", 1000),
              param.getDefault("solver_tol", 1e-8)),
              serialSolver_(param),
              output_to_file_(param.getDefault("output_to_file", false)),
              verbose_(param.getDefault("verbose", 0)),
              param_(param),
              max_iter_(param.getDefault("max_iter", 10)),
              abs_res_tol_(param.getDefault("abs_res_tol", 1e-6))
{
    wrapEquelleRuntimeCUDA::init_cusparse();
}

// Destructor:
EquelleRuntimeCUDA::~EquelleRuntimeCUDA() 
{
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

CollOfScalar EquelleRuntimeCUDA::norm(const CollOfScalar& scalars) const
{
    return scalars.norm();
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

Scalar EquelleRuntimeCUDA::twoNorm(const CollOfScalar& vals) const {
    return std::sqrt( sumReduce(vals*vals) );
    // This should be implemented without having to multiply matrices.
}



void EquelleRuntimeCUDA::output(const String& tag, const double val) const
{
    std::cout << tag << " = " << val << std::endl;
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




void EquelleRuntimeCUDA::output(const String& tag, const CollOfScalar& coll)
{
    // Get data back to host
    std::vector<double> host = coll.copyToHost();
    
    if (output_to_file_) {
    // std::map<std::string, int> outputcount_;
    // set file name to tag-0000X.output
    int count = -1;
    auto it = outputcount_.find(tag);
    if ( it == outputcount_.end()) {
        count = 0;
        outputcount_[tag] = 1; // should contain the count to be used next time for same tag.
    } else {
        count = outputcount_[tag];
        ++outputcount_[tag];
    }
    std::ostringstream fname;
    fname << tag << "-" << std::setw(5) << std::setfill('0') << count << ".output";
    std::ofstream file(fname.str().c_str());
    if( !file ) {
        OPM_THROW(std::runtime_error, "Failed to open " << fname.str());
    }
    file.precision(16);
    std::copy(host.data(), host.data() + host.size(),
          std::ostream_iterator<double>(file, "\n"));
    } else {
    std::cout << "\n";
    std::cout << "Values in " << tag << std::endl;
    for(int i = 0; i < coll.size(); ++i) {
        std::cout << host[i] << "  ";
    }
    std::cout << std::endl;
    }
}


Scalar EquelleRuntimeCUDA::inputScalarWithDefault(const String& name,
                          const Scalar default_value) {
    return param_.getDefault(name, default_value);
}



CollOfScalar EquelleRuntimeCUDA::trinaryIf( const CollOfBool& predicate,
                                            const CollOfScalar& iftrue,
                                            const CollOfScalar& iffalse) const {
    // First, we need same size of all input
    if (iftrue.size() != iffalse.size() || iftrue.size() != predicate.size()) {
    OPM_THROW(std::runtime_error, "Collections are not of the same size");
    }
    // Call a wrapper which calls a kernel
    return trinaryIfWrapper(predicate, iftrue, iffalse);
}


CollOfScalar EquelleRuntimeCUDA::gradient( const CollOfScalar& cell_scalarfield ) const
{
    // This function is at the moment kept in order to be able to compare efficiency
    // against the new implementation, where we use the matrix from devOps_.

    // First, need cell_scalarfield to be defined on all cells:
    if ( cell_scalarfield.size() != dev_grid_.number_of_cells() ) {
    OPM_THROW(std::runtime_error, "Gradient needs input defined on AllCells()");
    }
    
    return gradientWrapper(cell_scalarfield,
                   dev_grid_.interiorFaces(),
                   dev_grid_.face_cells(),
                   devOps_);
}

CollOfScalar EquelleRuntimeCUDA::gradient_matrix( const CollOfScalar& cell_scalarfield ) const
{
    if ( cell_scalarfield.size() != dev_grid_.number_of_cells() ) {
        OPM_THROW(std::runtime_error, "Gradient needs input defined on AllCells()");
    }
    return devOps_.grad() * cell_scalarfield;
}

CollOfScalar EquelleRuntimeCUDA::divergence(const CollOfScalar& face_fluxes) const {
    
    // If the size is not the same as the number of faces, then the input is
    // given as interiorFaces. Then it has to be extended to AllFaces.
    if ( face_fluxes.size() != dev_grid_.number_of_faces() ) {
        CollOfFace int_faces = interiorFaces();
        // Extend to AllFaces():
        CollOfScalar allFluxes = operatorExtend(face_fluxes, int_faces, allFaces());
        return divergenceWrapper(allFluxes,
                                 dev_grid_,
                                 devOps_);
    } else {
        // We are on allFaces already, so let's go!
        return divergenceWrapper(face_fluxes,
                                 dev_grid_,
                                 devOps_); 
    }
}

CollOfScalar EquelleRuntimeCUDA::multiplyAdd(const CollOfScalar& a, const CollOfScalar& b, const CollOfScalar& c)
{
    return a * b + c;
}

CollOfScalar EquelleRuntimeCUDA::divergence_matrix(const CollOfScalar& face_fluxes) const 
{
    
    // The input need to be defined on allFaces() or interiorFaces()
    if ( face_fluxes.size() != dev_grid_.number_of_faces() &&
         face_fluxes.size() != devOps_.num_int_faces() ) {
        OPM_THROW(std::runtime_error, "Input for divergence has to be on AllFaces or on InteriorFaces()");
    }
    
    if ( face_fluxes.size() == dev_grid_.number_of_faces() ) {
        // All faces
        return devOps_.fulldiv() * face_fluxes;
    }
    else { // on internal faces
        return devOps_.div() * face_fluxes;
    }
}


CollOfScalar EquelleRuntimeCUDA::negGradient(const CollOfScalar& cell_scalarField) const
{
    OPM_THROW(std::runtime_error, "Not yet implemented, as it is not generated by Equelle front-end...");
}

CollOfScalar EquelleRuntimeCUDA::interiorDivergence(const CollOfScalar& face_fluxes) const
{
    OPM_THROW(std::runtime_error, "Not yet implemented, as it is not generated by Equelle front-end...");
}



// SQRT
CollOfScalar EquelleRuntimeCUDA::sqrt(const CollOfScalar& x) const {

    return sqrtWrapper(x);
}


// ------------- REDUCTIONS --------------

Scalar EquelleRuntimeCUDA::minReduce(const CollOfScalar& x) const {
    return x.reduce(MIN);
}

Scalar EquelleRuntimeCUDA::maxReduce(const CollOfScalar& x) const {
    return x.reduce(MAX);
}

Scalar EquelleRuntimeCUDA::sumReduce(const CollOfScalar& x) const {
    return x.reduce(SUM);
}

Scalar EquelleRuntimeCUDA::prodReduce(const CollOfScalar& x) const {
    return x.reduce(PRODUCT);
}


// -------------- END REDUCTIONS ----------------


// Serial solver:
CollOfScalar EquelleRuntimeCUDA::serialSolveForUpdate(const CollOfScalar& residual) const 
{
    // Want to solve A*x=b, where A is residual.der, b = residual.val

    hostMat hostA = residual.derivative().toHost();
    std::vector<double> hostb = residual.value().copyToHost();
    std::vector<double> hostX(hostb.size(), 0.0);

    Opm::LinearSolverInterface::LinearSolverReport rep
    = serialSolver_.solve(hostA.rows, hostA.nnz,
                  &hostA.rowPtr[0], &hostA.colInd[0], &hostA.vals[0],
                  &hostb[0], &hostX[0]);
    if (!rep.converged) {
    OPM_THROW(std::runtime_error, "Serial linear solver failed to converge.");
    }
       
    return CollOfScalar(hostX);
}

