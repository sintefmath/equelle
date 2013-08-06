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
    int index;
};


/// Topological entity for cell.
struct Face
{
    int index;
};


/// Topological collections.
typedef std::vector<Cell> Cells;
typedef std::vector<Face> Faces;


/// Types from opm-autodiff and Eigen.
typedef AutoDiff::ForwardBlock<double> CollOfScalarsAD;
typedef CollOfScalarsAD::V CollOfScalars;
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
    Cells allCells() const;
    Faces allFaces() const;
    Faces interiorFaces() const;
    Cells firstCell(const Faces& faces) const;
    Cells secondCell(const Faces& faces) const;
    CollOfScalars area(const Faces& faces) const;
    CollOfScalars volume(const Cells& cells) const;
    CollOfScalars length(const CollOfVectors& vectors) const;
    CollOfVectors centroid(const Faces& faces) const;
    CollOfVectors centroid(const Cells& cells) const;

    /// Operators.
    CollOfScalarsAD negGradient(const CollOfScalarsAD& cell_scalarfield) const;
    CollOfScalarsAD divergence(const CollOfScalarsAD& face_fluxes) const;

    /// Solver function.
    CollOfScalarsAD newtonSolve(const ResidualComputerInterface& rescomp,
				const CollOfScalarsAD& u_initialguess) const;

    /// Output.
    static void output(const std::string& tag, double val);
    static void output(const std::string& tag, const CollOfScalars& vals);
    static void output(const std::string& tag, const CollOfScalarsAD& vals);

    /// Input.
    static CollOfScalars getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
							    const std::string& name,
							    const int size);

    /// Creating primary variables.
    static CollOfScalarsAD singlePrimaryVariable(const CollOfScalars& initial_values);

private:
    /// Solver helper.
    CollOfScalars solveForUpdate(const CollOfScalarsAD& residual) const;
    /// Norms.
    double norm(const CollOfScalars& vals) const;
    double norm(const CollOfScalarsAD& vals) const;

    Opm::GridManager grid_manager_;
    const UnstructuredGrid& grid_;
    HelperOps ops_;
    Opm::LinearSolverFactory linsolver_;
};

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
