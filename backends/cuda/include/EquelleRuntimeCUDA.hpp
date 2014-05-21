/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_HEADER_INCLUDED


//#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>

//#include <Eigen/Eigen>

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid/GridManager.hpp>
#include <vector>
#include <string>
#include <map>
#include <array>


// Including device code
// This should be independent from the rest of the host code
//      and especially from any c++11 code.
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"
#include "CollOfVector.hpp"
#include "DeviceGrid.hpp"
#include "equelleTypedefs.hpp"
#include "CudaMatrix.hpp"
#include "DeviceHelperOps.hpp"
#include "LinearSolver.hpp"



namespace equelleCUDA {



// Array Of {X} Collection Of Scalar:

std::array<CollOfScalar, 1> makeArray( const CollOfScalar& t );

std::array<CollOfScalar, 2> makeArray( const CollOfScalar& t1, 
				       const CollOfScalar& t2 );

std::array<CollOfScalar, 3> makeArray( const CollOfScalar& t1,
				       const CollOfScalar& t2,
				       const CollOfScalar& t3 );

std::array<CollOfScalar, 4> makeArray( const CollOfScalar& t1,
				       const CollOfScalar& t2,
				       const CollOfScalar& t3,
				       const CollOfScalar& t4 );





// The Equelle runtime class.
// Contains methods corresponding to Equelle built-ins to make
// it easy to generate C++ code for an Equelle program.
//! Equelle runtime class for the CUDA back-end
class EquelleRuntimeCUDA
{
public:
    /// Constructor.
    EquelleRuntimeCUDA(const Opm::parameter::ParameterGroup& param);

    /// Destructor:
    ~EquelleRuntimeCUDA();

    /// Topology and geometry related.
    CollOfCell allCells() const;
    CollOfCell boundaryCells() const;
    CollOfCell interiorCells() const;
    CollOfFace allFaces() const;
    CollOfFace boundaryFaces() const;
    CollOfFace interiorFaces() const;
    CollOfCell firstCell(CollOfFace faces) const;
    CollOfCell secondCell(CollOfFace faces) const;
    template <int codim>
    CollOfScalar norm(const CollOfIndices<codim>& set) const;
    CollOfScalar norm(const CollOfVector& vectors) const;
    template <int codim>
    CollOfVector centroid(const CollOfIndices<codim>& set) const;
    CollOfVector normal(const CollOfFace& faces) const;


    /// Operators and math functions.
    CollOfScalar dot(const CollOfVector& v1, const CollOfVector& v2) const;
    CollOfScalar negGradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalar interiorDivergence(const CollOfScalar& face_fluxes) const;
    
    // Operators and math functions havahol
    CollOfScalar gradient(const CollOfScalar& cell_scalarfield) const;
    CollOfScalar gradient_matrix(const CollOfScalar& cell_scalarfield) const;
    CollOfScalar divergence(const CollOfScalar& fluxes) const;  
    CollOfScalar divergence_matrix(const CollOfScalar& fluxes) const;
    template<int codim>
    CollOfBool isEmpty(const CollOfIndices<codim>& set) const;
    
    // EXTEND and ON operators
    template<int codim>
    CollOfScalar operatorExtend(const CollOfScalar& data_in,
				const CollOfIndices<codim>& from_set,
				const CollOfIndices<codim>& to_set) const;
    
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
    
    CollOfScalar trinaryIf( const CollOfBool& predicate,
			    const CollOfScalar& iftrue,
			    const CollOfScalar& iffalse) const;
    
    template <int codim>
    CollOfIndices<codim> trinaryIf( const CollOfBool& predicate,
				    const CollOfIndices<codim>& iftrue,
				    const CollOfIndices<codim>& iffalse) const;
    
    /// Reductions.
    Scalar minReduce(const CollOfScalar& x) const;
    Scalar maxReduce(const CollOfScalar& x) const;
    Scalar sumReduce(const CollOfScalar& x) const;
    Scalar prodReduce(const CollOfScalar& x) const;
    
    // Special functions:
    CollOfScalar sqrt(const CollOfScalar& x) const;
    
    
    template <class ResidualFunctor> 
    CollOfScalar newtonSolve(const ResidualFunctor& rescomp,
			     const CollOfScalar& u_initialguess);
    
    //    template <int Num>
    //    std::array<CollOfScalarCPU, Num> newtonSolveSystem(const std::array<typename ResCompType<Num>::type, Num>& rescomp,
    //                                                    const std::array<CollOfScalarCPU, Num>& u_initialguess);
    
    /// Output.
    void output(const String& tag, Scalar val) const;
    void output(const String& tag, const CollOfScalar& coll);
    
    /// Input.
    Scalar inputScalarWithDefault(const String& name,
				  const Scalar default_value);
    template <class SomeCollection>
    CollOfScalar inputCollectionOfScalar(const String& name,
					 const SomeCollection& coll);
    
    template <int codim>
    CollOfIndices<codim> inputDomainSubsetOf( const String& name,
					      CollOfIndices<codim> superset);
    
    template <int codim>
    CollOfScalar inputCollectionOfScalar(const String& name,
						      const CollOfIndices<codim>& coll);
    
    SeqOfScalar inputSequenceOfScalar(const String& name);
    
    
    /// Ensuring requirements that may be imposed by Equelle programs.
    void ensureGridDimensionMin(const int minimum_grid_dimension) const;

    
    // ------- FUNCTIONS ONLY FOR TESTING -----------------------

    // Havahol - add a function to return grid in order to do testing here.
    UnstructuredGrid getGrid() const;
    
    CudaMatrix getGradMatrix() const { return devOps_.grad();};
    CudaMatrix getDivMatrix() const { return devOps_.div();};
    CudaMatrix getFulldivMatrix() const {return devOps_.fulldiv();};

    Scalar twoNormTester(const CollOfScalar& val) const { return twoNorm(val); };

    // ------------ PRIVATE MEMBERS -------------------------- //
private:
      
    /// Norms.
    Scalar twoNorm(const CollOfScalar& vals) const;
    
    /// Data members.
    std::unique_ptr<Opm::GridManager> grid_manager_;
    const UnstructuredGrid& grid_;
    equelleCUDA::DeviceGrid dev_grid_;
    mutable DeviceHelperOps devOps_;
    LinearSolver solver_;
    Opm::LinearSolverFactory serialSolver_;
    bool output_to_file_;
    int verbose_;
    const Opm::parameter::ParameterGroup& param_;
    std::map<std::string, int> outputcount_;
    // For newtonSolve().
    int max_iter_;
    double abs_res_tol_;

    CollOfScalar serialSolveForUpdate(const CollOfScalar& residual) const;


};

} // namespace equelleCUDA


// Include the implementations of template members.
#include "EquelleRuntimeCUDA_impl.hpp"

#endif // EQUELLERUNTIMECUDA_HEADER_INCLUDED
