#pragma once

#include <opm/autodiff/AutoDiffBlock.hpp>

#include <vector>
#include <string>

namespace equelle {

/// Codes for inner and outer boundaries
enum Boundary { outer = -1, inner = -2 };

/// Topological entities.r
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

// Binary ops for SeqOfScalar
inline SeqOfScalar operator*(const SeqOfScalar& sos, Scalar s)
{
    SeqOfScalar res = sos;
    for (Scalar& r : res) {
        r *= s;
    }
    return res;
}
inline SeqOfScalar operator*(Scalar s, const SeqOfScalar& sos)
{
    return sos * s;
}
inline SeqOfScalar operator/(const SeqOfScalar& sos, Scalar s)
{
    return sos * (Scalar(1)/s);
}

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
inline CollOfBool operator<(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() < y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<=(const Scalar& s, const CollOfScalar& x)
{
    return s <= x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<=(const CollOfScalar& x, const Scalar& s)
{
    return x.value() <= s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator<=(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() <= y.value();
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
inline CollOfBool operator>(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() > y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>=(const Scalar& s, const CollOfScalar& x)
{
    return s >= x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>=(const CollOfScalar& x, const Scalar& s)
{
    return x.value() >= s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator>=(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() >= y.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator==(const Scalar& s, const CollOfScalar& x)
{
    return s == x.value();
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator==(const CollOfScalar& x, const Scalar& s)
{
    return x.value() == s;
}

/// This operator is not provided by AutoDiffBlock, so we must add it here.
inline CollOfBool operator==(const CollOfScalar& x, const CollOfScalar& y)
{
    return x.value() == y.value();
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


/// A helper type for ensuring AutoDiffBlock objects are converted
/// to CollOfScalar when necessary for template functions.
template <class Coll>
struct CollType { typedef Coll Type; };
template<>
struct CollType<Opm::AutoDiffBlock<double>> { typedef CollOfScalar Type; };


/// Simplify support of array literals.
// template <typename T>
// std::array<typename CollType<T>::Type, 1> makeArray(const T& t)
// {
//     return std::array<typename CollType<T>::Type,1>{{t}};
// }
template <typename T>
std::tuple<typename CollType<T>::Type> makeArray(const T& t)
{
    return std::tuple<typename CollType<T>::Type>{t};
}
// template <typename T1, typename T2>
// std::array<typename CollType<T1>::Type, 2> makeArray(const T1& t1, const T2& t2)
// {
//     return std::array<typename CollType<T1>::Type,2>{{t1, t2}};
// }
template <typename T1, typename T2>
std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type> makeArray(const T1& t1, const T2& t2)
{
    return std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type>{t1, t2};
}
// template <typename T1, typename T2, typename T3>
// std::array<typename CollType<T1>::Type, 3> makeArray(const T1& t1, const T2& t2, const T3& t3)
// {
//     return std::array<typename CollType<T1>::Type,3>{{t1, t2, t3}};
// }
template <typename T1, typename T2, typename T3>
std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type, typename CollType<T3>::Type>
makeArray(const T1& t1, const T2& t2, const T3& t3)
{
    return std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type, typename CollType<T3>::Type>
        {t1, t2, t3};
}
// template <typename T1, typename T2, typename T3, typename T4>
// std::array<typename CollType<T1>::Type, 4> makeArray(const T1& t1, const T2& t2, const T3& t3, const T4& t4)
// {
//     return std::array<typename CollType<T1>::Type,4>{{t1, t2, t3, t4}};
// }
template <typename T1, typename T2, typename T3, typename T4>
std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type, typename CollType<T3>::Type, typename CollType<T4>::Type>
makeArray(const T1& t1, const T2& t2, const T3& t3, const T4& t4)
{
    return std::tuple<typename CollType<T1>::Type, typename CollType<T2>::Type, typename CollType<T3>::Type, typename CollType<T4>::Type>

        {t1, t2, t3, t4};
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

} // namespace equelle
