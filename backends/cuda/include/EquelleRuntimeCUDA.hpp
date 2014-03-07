/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HEADER_INCLUDED


//#include <opm/autodiff/AutoDiffBlock.hpp>
//#include <opm/autodiff/AutoDiffHelpers.hpp>

//#include <Eigen/Eigen>

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <vector>
#include <string>
#include <map>

#include <thrust/device_vector.h>



// Including device code
// This should be independent from the rest of the host code
//      and especially from any c++11 code.
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"
#include "DeviceGrid.hpp"
#include "equelleTypedefs.hpp"

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
typedef std::vector<Cell> CollOfCellCPU;
typedef std::vector<Face> CollOfFaceCPU;

// Basic types. Note that we do not have Vector type defined
// although the CollOfVector type is.
typedef double Scalar;
typedef bool Bool;
typedef std::string String;

// Collections and sequences (apart from Collection Of Scalar).
//typedef Eigen::Array<bool, Eigen::Dynamic, 1> CollOfBool;
//typedef std::vector<bool> CollOfBool;
typedef std::vector<Scalar> SeqOfScalar;


/// The Collection Of Scalar type is based on Eigen and opm-autodiff.
/// It uses inheritance to provide extra interfaces for ease of use,
/// notably converting constructors.
class CollOfScalarCPU : public std::vector<double>
{
public:
    typedef std::vector<double> ADB; //Opm::AutoDiffBlock<double> ADB;
    //typedef ADB::V V;
    CollOfScalarCPU()
        : ADB(ADB())
    {
    }
    CollOfScalarCPU(const ADB& adb)
    : ADB(adb)
    {
    }

};    



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
    CollOfCell firstCell(equelleCUDA::CollOfFace faces) const;
    CollOfCell secondCell(equelleCUDA::CollOfFace faces) const;
    CollOfScalarCPU norm(const CollOfFaceCPU& faces) const;
    CollOfScalarCPU norm(const CollOfCellCPU& cells) const;
    CollOfScalarCPU norm(const CollOfVector& vectors) const;
    CollOfVector centroid(const CollOfFaceCPU& faces) const;
    CollOfVector centroid(const CollOfCellCPU& cells) const;
    CollOfVector normal(const CollOfFaceCPU& faces) const;

    /// Operators and math functions.
    CollOfScalarCPU sqrt(const CollOfScalarCPU& x) const;
    CollOfScalarCPU dot(const CollOfVector& v1, const CollOfVector& v2) const;
    CollOfScalarCPU gradient(const CollOfScalarCPU& cell_scalarfield) const;
    CollOfScalarCPU negGradient(const CollOfScalarCPU& cell_scalarfield) const;
    CollOfScalarCPU divergence(const CollOfScalarCPU& face_fluxes) const;
    CollOfScalarCPU interiorDivergence(const CollOfScalarCPU& face_fluxes) const;
    CollOfBool isEmpty(const CollOfCellCPU& cells) const;
    CollOfBool isEmpty(const CollOfFaceCPU& faces) const;
    template<int codim>
    CollOfBool isEmpty(const CollOfIndices<codim>& set) const;
    
    // EXTEND and ON operators
    template<int codim>
    CollOfScalar operatorExtend(const CollOfScalar& data_in,
				const CollOfIndices<codim>& from_set,
				const CollOfIndices<codim>& to_set);

    template<int codim>
    CollOfScalar operatorExtend(const Scalar& data, const CollOfIndices<codim>& set);

    template<int codim>
    CollOfScalar operatorOn(const CollOfScalar& data_in,
			    const CollOfIndices<codim>& from_set,
			    const CollOfIndices<codim>& to_set);

    // Implementation of the Equelle keyword On for CollOfIndices<>
    template<int codim_data, int codim_set>
    CollOfIndices<codim_data> operatorOn( const CollOfIndices<codim_data>& in_data,
					  const CollOfIndices<codim_set>& from_set,
					  const CollOfIndices<codim_set>& to_set);


    //template <class SomeCollection1, class SomeCollection2>
    //typename CollType<SomeCollection1>::Type
    //trinaryIf(const CollOfBool& predicate,
    //          const SomeCollection1& iftrue,
    //          const SomeCollection2& iffalse) const;

    /// Reductions.
    Scalar minReduce(const CollOfScalarCPU& x) const;
    Scalar maxReduce(const CollOfScalarCPU& x) const;
    Scalar sumReduce(const CollOfScalarCPU& x) const;
    Scalar prodReduce(const CollOfScalarCPU& x) const;

    /// Solver functions.
    template <class ResidualFunctor>
    CollOfScalarCPU newtonSolve(const ResidualFunctor& rescomp,
                             const CollOfScalarCPU& u_initialguess);

//    template <int Num>
//    std::array<CollOfScalarCPU, Num> newtonSolveSystem(const std::array<typename ResCompType<Num>::type, Num>& rescomp,
//                                                    const std::array<CollOfScalarCPU, Num>& u_initialguess);

    /// Output.
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalarCPU& vals);
    void output(const String& tag, const equelleCUDA::CollOfScalar& coll);

    /// Input.
    Scalar inputScalarWithDefault(const String& name,
                                          const Scalar default_value);
    CollOfFaceCPU inputDomainSubsetOf(const String& name,
                                   const CollOfFaceCPU& superset);
    CollOfCellCPU inputDomainSubsetOf(const String& name,
                                   const CollOfCellCPU& superset);
    template <class SomeCollection>
    equelleCUDA::CollOfScalar inputCollectionOfScalar(const String& name,
                                         const SomeCollection& coll);

    // input havahol
    template <int codim>
    equelleCUDA::CollOfIndices<codim> inputDomainSubsetOf( const String& name,
							 equelleCUDA::CollOfIndices<codim> superset);
	
    // input havahol
    template <int codim>
    equelleCUDA::CollOfScalar inputCollectionOfScalar(const String& name,
						      const equelleCUDA::CollOfIndices<codim>& coll);
										 
    SeqOfScalar inputSequenceOfScalar(const String& name);


    /// Ensuring requirements that may be imposed by Equelle programs.
    void ensureGridDimensionMin(const int minimum_grid_dimension) const;

    // Havahol - add a function to return grid in order to do testing here.
    UnstructuredGrid getGrid() const;

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
    equelleCUDA::DeviceGrid dev_grid_;
    //Opm::HelperOps ops_;
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

#endif // EQUELLERUNTIMECUDA_HEADER_INCLUDED
