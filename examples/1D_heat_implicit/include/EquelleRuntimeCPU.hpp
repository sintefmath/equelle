/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECPU_HEADER_INCLUDED
#define EQUELLERUNTIMECPU_HEADER_INCLUDED


#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <vector>
#include <string>


/// Topological entity for cell.
struct Cell
{
    Cell(void): index_(-1) {}
    Cell(const int index): index_(index) {}
    int index_;
};


/// Topological entity for cell.
struct Face
{
    int index_;
};


/// Topological collections.
typedef std::vector<Cell> CollOfCells;
typedef std::vector<Face> CollOfFaces;


/// Types from opm-autodiff and Eigen.
typedef AutoDiff::ForwardBlock<double> CollOfScalarsAD;
typedef CollOfScalarsAD::V CollOfScalars;

class CollOfScalarsOnColl
{
public:
    // This constructor makes this class operate exactly like the old one, as a fallback solution if we do not want to be strict about the "On-ness".
    CollOfScalarsOnColl(const CollOfScalars &c)
    {
        coll_ = c;
        on_collection_ = 0;
    }
    template<typename ONCOLL>
    CollOfScalarsOnColl(const CollOfScalars &c, const ONCOLL &oncoll)
    {
        coll_ = c;
        on_collection_ = &oncoll;
    }
    CollOfScalarsOnColl operator-(const CollOfScalarsOnColl &rhs) const
    {
        if ( on_collection_ != rhs.getOnColl() ) {
            std::cout << "\n\nKABOOM!!! This should not be allowed... Trying to subtract Collection Of Scalar *On* different Collections!\n\n" << std::endl;
        }
        return CollOfScalarsOnColl( coll_ - rhs.coll_, 0 );
    }
    CollOfScalars getColl(void) const
    {
        return coll_;
    }
    const void *getOnColl(void) const
    {
        return on_collection_;
    }
private:
    CollOfScalars coll_;
    const void *on_collection_;
};

typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CollOfVectors;



/// Interface for residual computer class.
class ResidualComputerInterface
{
public:
    virtual CollOfScalarsAD compute(const CollOfScalarsAD& u) const = 0;
};



/// The Equelle runtime class.
/// Contains methods corresponding to Equelle built-ins to make
/// it easy to generate C++ code for an Equelle program.
class EquelleRuntimeCPU
{
public:
    /// Constructor.
    EquelleRuntimeCPU(const Opm::parameter::ParameterGroup& param);

    /// Topology and geometry related.
    CollOfCells allCells() const;
    CollOfCells boundaryCells() const;
    CollOfCells interiorCells() const;
    CollOfFaces allFaces() const;
    CollOfFaces boundaryFaces() const;
    CollOfFaces interiorFaces() const;
    CollOfCells firstCell(const CollOfFaces& faces) const;
    CollOfCells secondCell(const CollOfFaces& faces) const;
    CollOfScalars norm(const CollOfFaces& faces) const;
    CollOfScalars norm(const CollOfCells& cells) const;
    CollOfScalars norm(const CollOfVectors& vectors) const;
    CollOfVectors centroid(const CollOfFaces& faces) const;
    CollOfVectors centroid(const CollOfCells& cells) const;

    /// Operators.
    CollOfScalarsAD negGradient(const CollOfScalarsAD& cell_scalarfield) const;
    CollOfScalarsAD divergence(const CollOfScalarsAD& face_fluxes) const;

    /// Solver function.
    CollOfScalarsAD newtonSolve(const ResidualComputerInterface& rescomp,
				const CollOfScalarsAD& u_initialguess) const;

    /// Output.
    static void output(const std::string& tag, double val);
    static void output(const std::string& tag, const CollOfScalars& vals);
    static void output(const std::string& tag, const CollOfScalarsOnColl& vals);
    static void output(const std::string& tag, const CollOfScalarsAD& vals);

    /// Input.
    static CollOfScalars getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
							    const std::string& name,
							    const int size);

    /// Creating primary variables.
    static CollOfScalarsAD singlePrimaryVariable(const CollOfScalars& initial_values);

private:
    /// Topology helpers
    bool boundaryCell(const int cell_index) const;

    /// Solver helper.
    CollOfScalars solveForUpdate(const CollOfScalarsAD& residual) const;
    /// Norms.
    double twoNorm(const CollOfScalars& vals) const;
    double twoNorm(const CollOfScalarsAD& vals) const;

    Opm::GridManager grid_manager_;
    const UnstructuredGrid& grid_;
    HelperOps ops_;
    Opm::LinearSolverFactory linsolver_;
};

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
