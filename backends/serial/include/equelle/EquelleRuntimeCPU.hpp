/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#pragma once

#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>

#include <vector>
#include <string>
#include <map>

#include "equelle/equelleTypes.hpp"

namespace equelle {

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
    CollOfVector normal(const CollOfFace& faces) const;

    /// Operators and math functions.
    CollOfScalar sqrt(const CollOfScalar& x) const;
    CollOfScalar dot(const CollOfVector& v1, const CollOfVector& v2) const;
    CollOfScalar gradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalar negGradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalar divergence(const CollOfScalar& face_fluxes) const;
    CollOfScalar interiorDivergence(const CollOfScalar& face_fluxes) const;
    CollOfBool isEmpty(const CollOfCell& cells) const;
    CollOfBool isEmpty(const CollOfFace& faces) const;

    template <class EntityCollection>
    CollOfScalar operatorExtend(const Scalar data, const EntityCollection& to_set);

    template <class SomeCollection, class EntityCollection>
    SomeCollection operatorExtend(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection, class EntityCollection>
    typename CollType<SomeCollection>::Type operatorOn(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection1, class SomeCollection2>
    typename CollType<SomeCollection1>::Type
    trinaryIf(const CollOfBool& predicate,
              const SomeCollection1& iftrue,
              const SomeCollection2& iffalse) const;

    /// Reductions.
    Scalar minReduce(const CollOfScalar& x) const;
    Scalar maxReduce(const CollOfScalar& x) const;
    Scalar sumReduce(const CollOfScalar& x) const;
    Scalar prodReduce(const CollOfScalar& x) const;

    /// Solver functions.
    template <class ResidualFunctor>
    CollOfScalar newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalar& u_initialguess);

    template <int Num>
    std::array<CollOfScalar, Num> newtonSolveSystem(const std::array<typename ResCompType<Num>::type, Num>& rescomp,
                                                    const std::array<CollOfScalar, Num>& u_initialguess);

    /// Output.
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalar& vals);

    /// Input.
    Scalar inputScalarWithDefault(const String& name,
                                          const Scalar default_value);
    CollOfFace inputDomainSubsetOf(const String& name,
                                   const CollOfFace& superset);
    CollOfCell inputDomainSubsetOf(const String& name,
                                   const CollOfCell& superset);
    template <class SomeCollection>
    CollOfScalar inputCollectionOfScalar(const String& name,
                                         const SomeCollection& coll);

    SeqOfScalar inputSequenceOfScalar(const String& name);


    /// Ensuring requirements that may be imposed by Equelle programs.
    void ensureGridDimensionMin(const int minimum_grid_dimension) const;

private:
    /// Topology helpers
    bool boundaryCell(const int cell_index) const;

    /// Creating primary variables.
    static CollOfScalar singlePrimaryVariable(const CollOfScalar& initial_values);

    /// Solver helper.
    CollOfScalar solveForUpdate(const CollOfScalar& residual) const;

    /// Norms.
    Scalar twoNorm(const CollOfScalar& vals) const;

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


Opm::GridManager* createGridManager(const Opm::parameter::ParameterGroup& param);


} // namespace equelle

// Include the implementations of template members.
#include "EquelleRuntimeCPU_impl.hpp"


