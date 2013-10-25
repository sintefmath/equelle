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
#include <map>


/// Topological entities.
template <int Codim>
struct TopologicalEntity
{
    TopologicalEntity() : index(-1) {}
    explicit TopologicalEntity(const int ind) : index(ind) {}
    int index;
    bool operator<(const TopologicalEntity& t) const { return index < t.index; }
    bool operator==(const TopologicalEntity& t) const { return index == t.index; }
};

/// Topological entity for cell.
typedef TopologicalEntity<0> Cell;

/// Topological entity for cell.
typedef TopologicalEntity<1> Face;

/// Topological collections.
typedef std::vector<Cell> CollOfCell;
typedef std::vector<Face> CollOfFace;

// Basic types. Note that we do not have Vector type defined
// although the CollOfVector type is.
typedef double Scalar;
typedef bool Bool;
typedef std::string String;

/// Types from opm-autodiff and Eigen.

// The following trick lets us automatically convert CollOfScalar to CollOfScalarAD
//typedef Opm::AutoDiffBlock<Scalar> CollOfScalarAD;
typedef Opm::AutoDiffBlock<Scalar> ADB;
struct CollOfScalarAD : public ADB
{
    CollOfScalarAD(const ADB& adb)
        : ADB(adb)
    {
    }
    CollOfScalarAD(const ADB::V& x)
        : ADB(ADB::constant(x))
    {
    }
};
typedef ADB::V CollOfScalar;
typedef Eigen::Array<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> CollOfVector;
typedef Eigen::Array<bool, Eigen::Dynamic, 1> CollOfBool;
typedef std::vector<Scalar> SeqOfScalar;



/// The Equelle runtime class.
/// Contains methods corresponding to Equelle built-ins to make
/// it easy to generate C++ code for an Equelle program.
class EquelleRuntimeCPU
{
public:
    /// Constructor.
    EquelleRuntimeCPU(const Opm::parameter::ParameterGroup& param);

    /// Topology and geometry related.
    CollOfCell allCells() const;
    CollOfCell boundaryCells() const;
    CollOfCell interiorCells() const;
    CollOfFace allFaces() const;
    CollOfFace boundaryFaces() const;
    CollOfFace interiorFaces() const;
    CollOfCell firstCell(const CollOfFace& faces) const;
    CollOfCell secondCell(const CollOfFace& faces) const;
    CollOfScalar norm(const CollOfFace& faces) const;
    CollOfScalar norm(const CollOfCell& cells) const;
    CollOfScalar norm(const CollOfVector& vectors) const;
    CollOfVector centroid(const CollOfFace& faces) const;
    CollOfVector centroid(const CollOfCell& cells) const;

    /// Operators.
    CollOfScalar gradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalarAD gradient(const CollOfScalarAD& cell_scalarfield) const;
    CollOfScalar negGradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalarAD negGradient(const CollOfScalarAD& cell_scalarfield) const;
    CollOfScalar divergence(const CollOfScalar& face_fluxes) const;
    CollOfScalarAD divergence(const CollOfScalarAD& face_fluxes) const;
    CollOfScalar interiorDivergence(const CollOfScalar& face_fluxes) const;
    CollOfScalarAD interiorDivergence(const CollOfScalarAD& face_fluxes) const;
    CollOfBool isEmpty(const CollOfCell& cells) const;
    CollOfBool isEmpty(const CollOfFace& faces) const;
    template <class EntityCollection>
    CollOfScalar operatorOn(const Scalar data, const EntityCollection& to_set);
    template <class SomeCollection, class EntityCollection>
    SomeCollection operatorOn(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection>
    SomeCollection trinaryIf(const CollOfBool& predicate,
                             const SomeCollection& iftrue,
                             const SomeCollection& iffalse) const;

    /// Solver function.
    template <class ResidualFunctor>
    CollOfScalar newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalar& u_initialguess);

    /// Output.
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalar& vals);
    void output(const String& tag, const CollOfScalarAD& vals);

    /// Input.
    Scalar userSpecifiedScalarWithDefault(const String& name,
                                          const Scalar default_value);
    CollOfFace userSpecifiedCollectionOfFaceSubsetOf(const String& name,
                                                     const CollOfFace& superset);
    template <class SomeCollection>
    CollOfScalar userSpecifiedCollectionOfScalar(const String& name,
                                                 const SomeCollection& coll);

    SeqOfScalar userSpecifiedSequenceOfScalar(const String& name);

private:
    /// Topology helpers
    bool boundaryCell(const int cell_index) const;

    /// Creating primary variables.
    static CollOfScalarAD singlePrimaryVariable(const CollOfScalar& initial_values);

    /// Solver helper.
    CollOfScalar solveForUpdate(const CollOfScalarAD& residual) const;

    /// Norms.
    Scalar twoNorm(const CollOfScalar& vals) const;
    Scalar twoNorm(const CollOfScalarAD& vals) const;

    /// Data members.
    std::unique_ptr<Opm::GridManager> grid_manager_;
    const UnstructuredGrid& grid_;
    Opm::HelperOps ops_;
    Opm::LinearSolverFactory linsolver_;
    bool output_to_file_;
    int verbose_;
    const Opm::parameter::ParameterGroup& param_;
    std::map<std::string, int> outputcount_;
    // For newtonSolve().
    int max_iter_;
    double abs_res_tol_;
};

// Include the implementations of template members.
#include "EquelleRuntimeCPU_impl.hpp"

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
