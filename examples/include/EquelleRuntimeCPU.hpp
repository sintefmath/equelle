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

// Collections and sequences (apart from Collection Of Scalar).
typedef Eigen::Array<bool, Eigen::Dynamic, 1> CollOfBool;
typedef std::vector<Scalar> SeqOfScalar;

/// The Collection Of Scalar type is based on Eigen and opm-autodiff.
/// It uses inheritance to provide extra interfaces for ease of use,
/// notably converting constructors.
class CollOfScalar : public Opm::AutoDiffBlock<double>
{
public:
    typedef Opm::AutoDiffBlock<double> ADB;
    //typedef ADB::V V;
    CollOfScalar()
        : ADB(ADB::null())
    {
    }
    CollOfScalar(const ADB& adb)
        : ADB(adb)
    {
    }
    CollOfScalar(const ADB::V& x)
        : ADB(ADB::constant(x))
    {
    }
};

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalar operator-(const CollOfScalar& x)
{
    return CollOfScalar::V::Zero(x.size()) - x;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalar operator/(const Scalar& s, const CollOfScalar& x)
{
    return CollOfScalar::V::Constant(x.size(), s) / x;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalar operator/(const CollOfScalar& x, const Scalar& s)
{
    return x / CollOfScalar::V::Constant(x.size(), s);
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<(const Scalar& s, const CollOfScalar& x)
{
    return s < x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
        inline CollOfBool operator<(const CollOfScalar& x, const Scalar& s)
{
    return x.value() < s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const Scalar& s, const CollOfScalar& x)
{
    return s > x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const CollOfScalar& x, const Scalar& s)
{
    return x.value() > s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() < y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() > y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>=(const CollOfScalar& x, const Scalar& s)
{
    return x.value() >= s;
}

/// This function is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalar sqrt(const CollOfScalar& x)
{
    // d(sqrt(x))/dy = 1/(2*sqrt(x)) * dx/dy

    const auto& xjac = x.derivative();
    if (xjac.empty()) {
        return CollOfScalar(sqrt(x.value()));
    }
    const int num_blocks = xjac.size();
    std::vector<CollOfScalar::M> jac(num_blocks);
    typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> D;
    const auto sqrt_x = sqrt(x.value());
    const D one_over_two_sqrt_x = (0.5/sqrt_x).matrix().asDiagonal();
    for (int block = 0; block < num_blocks; ++block) {
        jac[block] = one_over_two_sqrt_x * xjac[block];
    }
    return CollOfScalar::ADB::function(sqrt_x, jac);
}



class CollOfVector
{
public:
    explicit CollOfVector(const int columns)
        : v(columns)
    {
    }
    const CollOfScalar& col(const int c) const
    {
        return v[c];
    }
    CollOfScalar& col(const int c)
    {
        return v[c];
    }
    int numCols() const
    {
        return v.size();
    }
private:
    std::vector<CollOfScalar> v;
};

inline CollOfVector operator+(const CollOfVector& v1, const CollOfVector& v2)
{
    const int dim = v1.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = v1.col(d) + v2.col(d);
    }
    return res;
}

inline CollOfVector operator-(const CollOfVector& v1, const CollOfVector& v2)
{
    const int dim = v1.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = v1.col(d) - v2.col(d);
    }
    return res;
}

inline CollOfVector operator-(const CollOfVector& x)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = -x.col(d);
    }
    return res;
}

inline CollOfVector operator*(const CollOfVector& x, const Scalar& s)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = x.col(d) * s;
    }
    return res;
}

inline CollOfVector operator*(const CollOfVector& x, const CollOfScalar& s)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = x.col(d) * s;
    }
    return res;
}

inline CollOfVector operator*(const Scalar& s, const CollOfVector& x)
{
    return x * s; // Commutative.
}

inline CollOfVector operator*(const CollOfScalar& s, const CollOfVector& x)
{
    return x * s; // Commutative.
}

inline CollOfVector operator/(const CollOfVector& x, const Scalar& s)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = x.col(d) / s;
    }
    return res;
}

inline CollOfVector operator/(const CollOfVector& x, const CollOfScalar& s)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = x.col(d) / s;
    }
    return res;
}


/// Simplify support of array literals.
template <typename T>
std::array<T, 1> makeArray(const T& t)
{
    return std::array<T,1>{{t}};
}
template <typename T>
std::array<T, 2> makeArray(const T& t1, const T& t2)
{
    return std::array<T,2>{{t1, t2}};
}
template <typename T>
std::array<T, 3> makeArray(const T& t1, const T& t2, const T& t3)
{
    return std::array<T,3>{{t1, t2, t3}};
}
template <typename T>
std::array<T, 4> makeArray(const T& t1, const T& t2, const T& t3, const T& t4)
{
    return std::array<T,4>{{t1, t2, t3, t4}};
}

/// A helper type for newtonSolveSystem
template <int Num>
struct ResCompType;

template <>
struct ResCompType<1>
{
    typedef std::function<CollOfScalar(const CollOfScalar&)> type;
};

template <>
struct ResCompType<2>
{
    typedef std::function<CollOfScalar(const CollOfScalar&, const CollOfScalar&)> type;
};

template <>
struct ResCompType<3>
{
    typedef std::function<CollOfScalar(const CollOfScalar&, const CollOfScalar&, const CollOfScalar&)> type;
};

template <>
struct ResCompType<4>
{
    typedef std::function<CollOfScalar(const CollOfScalar&, const CollOfScalar&, const CollOfScalar&, const CollOfScalar&)> type;
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
    SomeCollection operatorOn(const SomeCollection& data, const EntityCollection& from_set, const EntityCollection& to_set);

    template <class SomeCollection>
    SomeCollection trinaryIf(const CollOfBool& predicate,
                             const SomeCollection& iftrue,
                             const SomeCollection& iffalse) const;

    /// Solver function.
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

// Include the implementations of template members.
#include "EquelleRuntimeCPU_impl.hpp"

#endif // EQUELLERUNTIMECPU_HEADER_INCLUDED
