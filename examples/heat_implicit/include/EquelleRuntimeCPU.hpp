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
typedef Eigen::Array<bool, Eigen::Dynamic, 1> CollOfBooleans;



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
    CollOfScalars gradient(const CollOfScalars& cell_scalarfield) const;
    CollOfScalarsAD gradient(const CollOfScalarsAD& cell_scalarfield) const;
    CollOfScalars negGradient(const CollOfScalars& cell_scalarfield) const;
    CollOfScalarsAD negGradient(const CollOfScalarsAD& cell_scalarfield) const;
    CollOfScalars divergence(const CollOfScalars& face_fluxes) const;
    CollOfScalarsAD divergence(const CollOfScalarsAD& face_fluxes) const;
    CollOfScalars interiorDivergence(const CollOfScalars& face_fluxes) const;
    CollOfScalarsAD interiorDivergence(const CollOfScalarsAD& face_fluxes) const;
    CollOfBooleans isEmpty(const CollOfCells& cells) const;
    CollOfBooleans isEmpty(const CollOfFaces& faces) const;
    template <class EntityCollection>
    CollOfScalars operatorOn(const double data, const EntityCollection& to_set);
    template <class SomeCollection, class EntityCollection>
    SomeCollection operatorOn(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection>
    SomeCollection trinaryIf(const CollOfBooleans& predicate,
                             const SomeCollection& iftrue,
                             const SomeCollection& iffalse) const;

    /// Solver function.
    template <class ResidualFunctor>
    CollOfScalarsAD newtonSolve(const ResidualFunctor& rescomp,
				const CollOfScalars& u_initialguess) const;

    /// Output.
    static void output(const std::string& tag, double val);
    static void output(const std::string& tag, const CollOfScalars& vals);
    static void output(const std::string& tag, const CollOfScalarsOnColl& vals);
    static void output(const std::string& tag, const CollOfScalarsAD& vals);

    /// Input.
    static CollOfFaces getUserSpecifiedCollectionOfFaceSubsetOf(const Opm::parameter::ParameterGroup& param,
                                                                const std::string& name,
                                                                const CollOfFaces& superset);
    static CollOfScalars getUserSpecifiedCollectionOfScalar(const Opm::parameter::ParameterGroup& param,
							    const std::string& name,
							    const int size);


private:
    /// Topology helpers
    bool boundaryCell(const int cell_index) const;

    /// Creating primary variables.
    static CollOfScalarsAD singlePrimaryVariable(const CollOfScalars& initial_values);

    /// Solver helper.
    CollOfScalars solveForUpdate(const CollOfScalarsAD& residual) const;

    /// Norms.
    double twoNorm(const CollOfScalars& vals) const;
    double twoNorm(const CollOfScalarsAD& vals) const;

    /// Data members.
    Opm::GridManager grid_manager_;
    const UnstructuredGrid& grid_;
    HelperOps ops_;
    Opm::LinearSolverFactory linsolver_;
};

// Include the implementations of template members.
#include "EquelleRuntimeCPU_impl.hpp"

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
