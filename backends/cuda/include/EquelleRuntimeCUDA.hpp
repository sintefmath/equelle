/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HEADER_INCLUDED


#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <vector>
#include <string>
#include <map>

// Including device code
// This should be independent from the rest of the host code
//      and especially from any c++11 code.
#include "EquelleRuntimeCUDA_cuda.hpp"

// Forward declarations for the Device types
//class CollOfScalar;



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
class CollOfScalarCPU : public Opm::AutoDiffBlock<double>
{
public:
    typedef Opm::AutoDiffBlock<double> ADB;
    //typedef ADB::V V;
    CollOfScalarCPU()
        : ADB(ADB::null())
    {
    }
    CollOfScalarCPU(const ADB& adb)
    : ADB(adb)
    {
    }
    CollOfScalarCPU(const ADB::V& x)
        : ADB(ADB::constant(x))
    {
    }
};    

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalarCPU operator-(const CollOfScalarCPU& x)
{
    return CollOfScalarCPU::V::Zero(x.size()) - x;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalarCPU operator/(const Scalar& s, const CollOfScalarCPU& x)
{
    return CollOfScalarCPU::V::Constant(x.size(), s) / x;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalarCPU operator/(const CollOfScalarCPU& x, const Scalar& s)
{
    return x / CollOfScalarCPU::V::Constant(x.size(), s);
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<(const Scalar& s, const CollOfScalarCPU& x)
{
    return s < x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<(const CollOfScalarCPU& x, const Scalar& s)
{
    return x.value() < s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const Scalar& s, const CollOfScalarCPU& x)
{
    return s > x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const CollOfScalarCPU& x, const Scalar& s)
{
    return x.value() > s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<(const CollOfScalarCPU& x, const CollOfScalarCPU& y)
{
    return x.value() < y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>(const CollOfScalarCPU& x, const CollOfScalarCPU& y)
{
    return x.value() > y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>=(const CollOfScalarCPU& x, const Scalar& s)
{
    return x.value() >= s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator==(const CollOfScalarCPU& x, const CollOfScalarCPU& y)
{
    return x.value() == y.value();
}

/// This function is not provided by AutoDiffBlock, so we must add it here.
inline CollOfScalarCPU sqrt(const CollOfScalarCPU& x)
{
    // d(sqrt(x))/dy = 1/(2*sqrt(x)) * dx/dy

    const auto& xjac = x.derivative();
    if (xjac.empty()) {
        return CollOfScalarCPU(sqrt(x.value()));
    }
    const int num_blocks = xjac.size();
    std::vector<CollOfScalarCPU::M> jac(num_blocks);
    typedef Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic> D;
    const auto sqrt_x = sqrt(x.value());
    const D one_over_two_sqrt_x = (0.5/sqrt_x).matrix().asDiagonal();
    for (int block = 0; block < num_blocks; ++block) {
        jac[block] = one_over_two_sqrt_x * xjac[block];
    }
    return CollOfScalarCPU::ADB::function(sqrt_x, jac);
}



class CollOfVector
{
public:
    explicit CollOfVector(const int columns)
        : v(columns)
    {
    }
    const CollOfScalarCPU& col(const int c) const
    {
        return v[c];
    }
    CollOfScalarCPU& col(const int c)
    {
        return v[c];
    }
    int numCols() const
    {
        return v.size();
    }
private:
    std::vector<CollOfScalarCPU> v;
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

inline CollOfVector operator*(const CollOfVector& x, const CollOfScalarCPU& s)
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

inline CollOfVector operator*(const CollOfScalarCPU& s, const CollOfVector& x)
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

inline CollOfVector operator/(const CollOfVector& x, const CollOfScalarCPU& s)
{
    const int dim = x.numCols();
    CollOfVector res(dim);
    for (int d = 0; d < dim; ++d) {
        res.col(d) = x.col(d) / s;
    }
    return res;
}



/// A helper type for ensuring AutoDiffBlock objects are converted
/// to CollOfScalarCPU when necessary for template functions.
template <class Coll>
struct CollType { typedef Coll Type; };
template<>
struct CollType<Opm::AutoDiffBlock<double>> { typedef CollOfScalarCPU Type; };


/// Simplify support of array literals.
template <typename T>
std::array<typename CollType<T>::Type, 1> makeArray(const T& t)
{
    return std::array<typename CollType<T>::Type,1>{{t}};
}
template <typename T1, typename T2>
std::array<typename CollType<T1>::Type, 2> makeArray(const T1& t1, const T2& t2)
{
    return std::array<typename CollType<T1>::Type,2>{{t1, t2}};
}
template <typename T1, typename T2, typename T3>
std::array<typename CollType<T1>::Type, 3> makeArray(const T1& t1, const T2& t2, const T3& t3)
{
    return std::array<typename CollType<T1>::Type,3>{{t1, t2, t3}};
}
template <typename T1, typename T2, typename T3, typename T4>
std::array<typename CollType<T1>::Type, 4> makeArray(const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
    return std::array<typename CollType<T1>::Type,4>{{t1, t2, t3, t4}};
}

/// A helper type for newtonSolveSystem
template <int Num>
struct ResCompType;

template <>
struct ResCompType<1>
{
    typedef std::function<CollOfScalarCPU(const CollOfScalarCPU&)> type;
};

template <>
struct ResCompType<2>
{
    typedef std::function<CollOfScalarCPU(const CollOfScalarCPU&, const CollOfScalarCPU&)> type;
};

template <>
struct ResCompType<3>
{
    typedef std::function<CollOfScalarCPU(const CollOfScalarCPU&, const CollOfScalarCPU&, const CollOfScalarCPU&)> type;
};

template <>
struct ResCompType<4>
{
    typedef std::function<CollOfScalarCPU(const CollOfScalarCPU&, const CollOfScalarCPU&, const CollOfScalarCPU&, const CollOfScalarCPU&)> type;
};


/// This class is copied from the class Span in opm-autodiff's AutoDiffHelpers.hpp,
/// since the version in the 2013.10 release has a minor misfeature:
/// the data members are const, which prevents copying and assignment.
/// Class name changed to ESpan to avoid conflict.
class ESpan
{
public:
    explicit ESpan(const int num)
    : num_(num),
      stride_(1),
      start_(0)
    {
    }
    ESpan(const int num, const int stride, const int start)
        : num_(num),
          stride_(stride),
          start_(start)
    {
    }
    int operator[](const int i) const
    {
        assert(i >= 0 && i < num_);
        return start_ + i*stride_;
    }
    int size() const
    {
        return num_;
    }


    class ESpanIterator
    {
    public:
        ESpanIterator(const ESpan* span, const int index)
            : span_(span),
              index_(index)
        {
        }
        ESpanIterator operator++()
        {
            ++index_;
            return *this;
        }
        ESpanIterator operator++(int)
        {
            ESpanIterator before_increment(*this);
            ++index_;
            return before_increment;
        }
        bool operator<(const ESpanIterator& rhs) const
        {
            assert(span_ == rhs.span_);
            return index_ < rhs.index_;
        }
        bool operator==(const ESpanIterator& rhs) const
        {
            assert(span_ == rhs.span_);
            return index_ == rhs.index_;
        }
        bool operator!=(const ESpanIterator& rhs) const
        {
            assert(span_ == rhs.span_);
            return index_ != rhs.index_;
        }
        int operator*()
        {
            return (*span_)[index_];
        }
    private:
        const ESpan* span_;
        int index_;
    };

    typedef ESpanIterator iterator;
    typedef ESpanIterator const_iterator;

    ESpanIterator begin() const
    {
        return ESpanIterator(this, 0);
    }

    ESpanIterator end() const
    {
        return ESpanIterator(this, num_);
    }

    bool operator==(const ESpan& rhs)
    {
        return num_ == rhs.num_ && start_ == rhs.start_ && stride_ == rhs.stride_;
    }

private:
    int num_;
    int stride_;
    int start_;
};



/// The Equelle runtime class.
/// Contains methods corresponding to Equelle built-ins to make
/// it easy to generate C++ code for an Equelle program.
class EquelleRuntimeCUDA
{
public:
    /// Constructor.
    EquelleRuntimeCUDA(const Opm::parameter::ParameterGroup& param);

    /// Topology and geometry related.
    CollOfCell allCells() const;
    CollOfCell boundaryCells() const;
    CollOfCell interiorCells() const;
    CollOfFace allFaces() const;
    CollOfFace boundaryFaces() const;
    CollOfFace interiorFaces() const;
    CollOfCell firstCell(const CollOfFace& faces) const;
    CollOfCell secondCell(const CollOfFace& faces) const;
    CollOfScalarCPU norm(const CollOfFace& faces) const;
    CollOfScalarCPU norm(const CollOfCell& cells) const;
    CollOfScalarCPU norm(const CollOfVector& vectors) const;
    CollOfVector centroid(const CollOfFace& faces) const;
    CollOfVector centroid(const CollOfCell& cells) const;
    CollOfVector normal(const CollOfFace& faces) const;

    /// Operators and math functions.
    CollOfScalarCPU sqrt(const CollOfScalarCPU& x) const;
    CollOfScalarCPU dot(const CollOfVector& v1, const CollOfVector& v2) const;
    CollOfScalarCPU gradient(const CollOfScalarCPU& cell_scalarfield) const;
    CollOfScalarCPU negGradient(const CollOfScalarCPU& cell_scalarfield) const;
    CollOfScalarCPU divergence(const CollOfScalarCPU& face_fluxes) const;
    CollOfScalarCPU interiorDivergence(const CollOfScalarCPU& face_fluxes) const;
    CollOfBool isEmpty(const CollOfCell& cells) const;
    CollOfBool isEmpty(const CollOfFace& faces) const;

    template <class EntityCollection>
    CollOfScalarCPU operatorExtend(const Scalar data, const EntityCollection& to_set);

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
    Scalar minReduce(const CollOfScalarCPU& x) const;
    Scalar maxReduce(const CollOfScalarCPU& x) const;
    Scalar sumReduce(const CollOfScalarCPU& x) const;
    Scalar prodReduce(const CollOfScalarCPU& x) const;

    /// Solver functions.
    template <class ResidualFunctor>
    CollOfScalarCPU newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalarCPU& u_initialguess);

    template <int Num>
    std::array<CollOfScalarCPU, Num> newtonSolveSystem(const std::array<typename ResCompType<Num>::type, Num>& rescomp,
                                                    const std::array<CollOfScalarCPU, Num>& u_initialguess);

    /// Output.
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalarCPU& vals);
    void output(const String& tag, const CollOfScalar& coll, int dummy);

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
    static CollOfScalarCPU singlePrimaryVariable(const CollOfScalarCPU& initial_values);

    /// Solver helper.
    CollOfScalarCPU solveForUpdate(const CollOfScalarCPU& residual) const;

    /// Norms.
    Scalar twoNorm(const CollOfScalarCPU& vals) const;

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
#include "EquelleRuntimeCUDA_impl.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"
#include "EquelleRuntimeCUDA_cuda.hpp"

#endif // EQUELLERUNTIMECUDA_HEADER_INCLUDED
