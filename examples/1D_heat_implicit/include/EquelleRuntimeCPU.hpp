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
typedef AutoDiff::ForwardBlock<double> ScalarsAD;
typedef ScalarsAD::V Scalars;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Vectors;


/// Interface for residual computer class.
class ResidualComputerInterface
{
public:
    virtual ScalarsAD compute(const ScalarsAD& u) const = 0;
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
    Faces internalFaces() const;
    Cells firstCell(const Faces& faces) const;
    Cells secondCell(const Faces& faces) const;
    Scalars area(const Faces& faces) const;
    Scalars volume(const Cells& cells) const;
    Scalars length(const Vectors& vectors) const;
    Vectors centroid(const Faces& faces) const;
    Vectors centroid(const Cells& cells) const;

    /// Operators.
    ScalarsAD negGradient(const ScalarsAD& cell_scalarfield) const;
    ScalarsAD divergence(const ScalarsAD& face_fluxes) const;

    /// Solver function.
    ScalarsAD newtonSolve(const ResidualComputerInterface& rescomp,
                          const ScalarsAD& u_initialguess) const;

    /// Output.
    static void output(const std::string& tag, double val);
    static void output(const std::string& tag, const Scalars& vals);
    static void output(const std::string& tag, const ScalarsAD& vals);

    /// Input.
    static Scalars getUserSpecifiedScalars(const Opm::parameter::ParameterGroup& param,
                                           const std::string& name,
                                           const int size);

    /// Creating primary variables.
    static ScalarsAD singlePrimaryVariable(const Scalars& initial_values);

private:
    /// Solver helper.
    Scalars solveForUpdate(const ScalarsAD& residual) const;
    /// Norms.
    double norm(const Scalars& vals) const;
    double norm(const ScalarsAD& vals) const;

    Opm::GridManager grid_manager_;
    const UnstructuredGrid& grid_;
    HelperOps ops_;
    Opm::LinearSolverFactory linsolver_;
};

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
