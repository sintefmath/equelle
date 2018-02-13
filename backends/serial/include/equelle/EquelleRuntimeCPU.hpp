/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#pragma once

#define SILENCE_EXTERNAL_WARNINGS

#include <opm/common/ErrorMacros.hpp>
#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/grid/UnstructuredGrid.h>
#include <opm/grid/GridManager.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>

#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <map>

#include "equelle/equelleTypes.hpp"

namespace equelle {

class StencilCollOfScalar;

/// The Equelle runtime class.
/// Contains methods corresponding to Equelle built-ins to make
/// it easy to generate C++ code for an Equelle program.
class EquelleRuntimeCPU
{
public:
    /// Constructor.
    EquelleRuntimeCPU( const Opm::ParameterGroup& param );
    EquelleRuntimeCPU( const UnstructuredGrid* grid, const Opm::ParameterGroup& param );

    /** @name Topology
     * Topology and geometry related. */
    ///@{
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
    ///@}

    /** @name Math
     * Operators and math functions. */
    ///@{
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
    ///@}


    /** @name Reductions. */
    ///@{
    Scalar minReduce(const CollOfScalar& x) const;
    Scalar maxReduce(const CollOfScalar& x) const;
    Scalar sumReduce(const CollOfScalar& x) const;
    Scalar prodReduce(const CollOfScalar& x) const;
    ///@}

    /// @name Solver functions.
    ///@{
    template <class ResidualFunctor>
    CollOfScalar newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalar& u_initialguess);


    template <class ... ResFuncs, class ... Colls>
    std::tuple<Colls...> newtonSolveSystem(const std::tuple<ResFuncs...>& rescomp,
                                           const std::tuple<Colls...>& u_initialguess);
    ///@}

    /// @name Output
    ///@{
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalar& vals);
    ///@}

    /// @name Input
    ///@{
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
    ///@}

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
    const Opm::ParameterGroup& param_;
    std::map<std::string, int> outputcount_;
    // For newtonSolve().
    int max_iter_;
    double abs_res_tol_;
};


Opm::GridManager* createGridManager(const Opm::ParameterGroup& param);


} // namespace equelle

// Include the implementations of template members.
#include "EquelleRuntimeCPU_impl.hpp"


