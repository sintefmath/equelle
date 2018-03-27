/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED


#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <opm/grid/utility/StopWatch.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "equelleTypedefs.hpp"
#include "wrapEquelleRuntime.hpp"

#include <thrust/host_vector.h>


using namespace equelleCUDA;






// -------------------------------------------------- //
// ------------------ INPUT ------------------------- //
// -------------------------------------------------- //


template <int codim>
equelleCUDA::CollOfScalar EquelleRuntimeCUDA::inputCollectionOfScalar( const String& name, const equelleCUDA::CollOfIndices<codim>& coll) 
{
    std::cout << "Copy from file " << name << std::endl;
    // Copy the reading part from the CPU back-end
    const int size = coll.size();
    const bool from_file = param_.getDefault(name + "_from_file", false);
    if ( from_file ) {
	const String filename = param_.get<String>(name + "_filename");
	std::ifstream is (filename.c_str());
	if (!is) {
	    OPM_THROW(std::runtime_error, "Could not find file " << filename);
	}
	std::istream_iterator<double> begin(is);
	std::istream_iterator<double> end;
	std::vector<double> data(begin, end);
	//CollOfScalar out(data);
	//out.setValuesFromFile(begin, end);
	if ( data.size() != size) {
	    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename); 
	}
	return equelleCUDA::CollOfScalar(data);
    }
    else {
	// There is a number in the parameter file
	//CollOfScalar out(size);
	//out.setValuesUniform(param_.get<double>(name));
	//out.setValuesUniform(param_.get<double>(name), size);
	return equelleCUDA::CollOfScalar(size, param_.get<double>(name));
    }
}



template <int codim>
equelleCUDA::CollOfIndices<codim> EquelleRuntimeCUDA::inputDomainSubsetOf(const String& name, equelleCUDA::CollOfIndices<codim> superset) 
{
    const String filename = param_.get<String>(name + "_filename");
    std::ifstream is(filename.c_str());
    if (!is) {
	OPM_THROW(std::runtime_error, "Could not find file " << filename);
    }
    std::istream_iterator<int> beg(is);
    std::istream_iterator<int> end;
    
    std::vector<int> std_vec(beg, end);
    if (!is_sorted(std_vec.begin(), std_vec.end()) ) {
	OPM_THROW(std::runtime_error, "Input set " << name << " was not sorted in ascending order");
    }
    
    thrust::host_vector<int> host(std_vec.begin(), std_vec.end());
    //thrust::device_vector<int> dev(std_vec.begin(), std_vec.end());
    //thrust::device_vector<int> dev(host.begin(), host.end());
    //thrust::sort(dev.begin(), dev.end());
      
    equelleCUDA::CollOfIndices<codim> subset(host);
    //subset.sort();
    
    // USING SORT ON THE DEVICE PRODUCES A STRANGE ERROR.
    // SORTING DONE ON THE HOST FOR NOW...
    // Believe this is because the _impl.hpp file is compiled by 
    // the gcc compiler as well through #includes
    
    superset.contains(subset, name);

    // The function works for correct sets without sort and contains,
    // but it also allows for illegal input.

    return subset;
}


// ---------------------------------------------------- //
// ----------------- GRID OPERATIONS ------------------ //
// ---------------------------------------------------- //


template <int codim>
CollOfScalar EquelleRuntimeCUDA::operatorExtend(const CollOfScalar& data_in,
						const CollOfIndices<codim>& from_set,
						const CollOfIndices<codim>& to_set) const {
    
    if (data_in.size() != from_set.size() ) {
	OPM_THROW(std::runtime_error, "data_in (size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in Extend function."); 
    }
    if (from_set.size() > to_set.size() ) {
	OPM_THROW(std::runtime_error, "From_set (size " << from_set.size() << ") has to be a subset of to_set (size " << to_set.size() << ")");
    }
    
    return dev_grid_.operatorExtend(data_in, from_set, to_set);
}

template <int codim>
CollOfScalar EquelleRuntimeCUDA::operatorExtend(const Scalar& data,
						const CollOfIndices<codim>& set) {
    return CollOfScalar(set.size(), data);
}


template <int codim>
CollOfScalar EquelleRuntimeCUDA::operatorOn(const CollOfScalar& data_in,
					    const CollOfIndices<codim>& from_set,
					    const CollOfIndices<codim>& to_set) {

    if ( data_in.size() != from_set.size()) {
	OPM_THROW(std::logic_error, "data_in (size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in On function for CollOfScalar.");
    }
    //if ( to_set.size() > from_set.size() ) {
    //	OPM_THROW(std::runtime_error, "To_set (size " << to_set.size() << ") has to be a subset of from_set (size " << from_set.size() << ")");
    //}

    return dev_grid_.operatorOn(data_in, from_set, to_set);
}



template<int codim_data, int codim_set>
CollOfIndices<codim_data> EquelleRuntimeCUDA::operatorOn( const CollOfIndices<codim_data>& data_in,
							  const CollOfIndices<codim_set>& from_set,
							  const CollOfIndices<codim_set>& to_set) {
    if ( data_in.size() != from_set.size() ) {
	OPM_THROW(std::logic_error, "data_in(size " << data_in.size() << ") and from_set (size " << from_set.size() << ") have to be of the same size in On function for CollOfIndices.");
    }
    return CollOfIndices<codim_data>(dev_grid_.operatorOn(data_in, from_set, to_set) );
}

// Is empty
template <int codim>
CollOfBool EquelleRuntimeCUDA::isEmpty(const CollOfIndices<codim>& set) const {
    return set.isEmpty();
}



// NORM

template <int codim>
CollOfScalar EquelleRuntimeCUDA::norm(const CollOfIndices<codim>& set) const {
    if (codim == 0) { // cells
	return dev_grid_.norm_of_cells(set.device_vector(), set.isFull());
    }
    else if (codim == 1) { // faces
	return dev_grid_.norm_of_faces(set.device_vector(), set.isFull());
    }
    else {
	OPM_THROW(std::runtime_error, "Norm of a Collection of Indices with codim " << codim << " is not supported.");
    }
}


// TRINARY IF 
template<int codim>
CollOfIndices<codim> EquelleRuntimeCUDA::trinaryIf( const CollOfBool& predicate,
						    const CollOfIndices<codim>& iftrue,
						    const CollOfIndices<codim>& iffalse) const
{
    if ( predicate.size() != iftrue.size() || predicate.size() != iffalse.size() ) {
	OPM_THROW(std::runtime_error, "The sets are not of the same size");
    }
    return CollOfIndices<codim>(wrapEquelleRuntimeCUDA::trinaryIfWrapper(predicate,
									 iftrue.device_vector(),
									 iffalse.device_vector()));
}




// CENTRIOD
template <int codim>
CollOfVector EquelleRuntimeCUDA::centroid( const CollOfIndices<codim>& set) const 
{
    if ( codim != 0 && codim != 1) {
	OPM_THROW(std::runtime_error, "Codim template parameter " << codim << " in EquelleRuntimeCUDA::centroid");
    }
    return dev_grid_.centroid(set.device_vector(), set.isFull(), codim);
}


// NEWTON SOLVE


template <class ResidualFunctor>
CollOfScalar EquelleRuntimeCUDA::newtonSolve(const ResidualFunctor& rescomp,
                                            const CollOfScalar& u_initialguess)
{
    Opm::time::StopWatch clock;
    clock.start();

    // Set up Newton loop.
 
    // Define the primary variable
    CollOfScalar u = CollOfScalar(u_initialguess, true);
 
    if (verbose_ > 2) {
        output("Initial u", u);
        output("    newtonSolve: norm (initial u)", twoNorm(u));
    }
    CollOfScalar residual = rescomp(u);   
    if (verbose_ > 2) {
        output("Initial residual", residual);
        output("    newtonSolve: norm (initial residual)", twoNorm(residual));
    }

    int iter = 0;

    // Debugging output not specified in Equelle.
    if (verbose_ > 1) {
        std::cout << "    newtonSolve: iter = " << iter << " (max = " << max_iter_
		  << "), norm(residual) = " << twoNorm(residual)
                  << " (tol = " << abs_res_tol_ << ")" << std::endl;
    }

    CollOfScalar du;

    // Execute newton loop until residual is small or we have used too many iterations.
    while ( (twoNorm(residual) > abs_res_tol_) && (iter < max_iter_) ) {
	
	if ( solver_.getSolver() == CPU ) {
	    du = serialSolveForUpdate(residual);
	}
	else {
	    // Solve linear equations for du, apply update.
	    du = solver_.solve(residual.derivative(),
			       residual.value(),
			       verbose_);
	}

	// du is a constant, hence, u is still a primary variable with an identity
	// matrix as its derivative.
	u = u - du;

        // Recompute residual.
        residual = rescomp(u);

        if (verbose_ > 2) {
            // Debugging output not specified in Equelle.
            output("u", u);
            output("    newtonSolve: norm(u)", twoNorm(u));
            output("residual", residual);
            output("    newtonSolve: norm(residual)", twoNorm(residual));
        }

        ++iter;

        // Debugging output not specified in Equelle.
        if (verbose_ > 1) {
            std::cout << "    newtonSolve: iter = " << iter << " (max = " << max_iter_
		      << "), norm(residual) = " << twoNorm(residual)
                      << " (tol = " << abs_res_tol_ << ")" << std::endl;
        }

    }
    if (verbose_ > 0) {
        if (twoNorm(residual) > abs_res_tol_) {
            std::cout << "Newton solver failed to converge in " << max_iter_ << " iterations" << std::endl;
        } else {
            std::cout << "Newton solver converged in " << iter << " iterations" << std::endl;
        }
    }

    if (verbose_ > 1) {
        std::cout << "Newton solver took: " << clock.secsSinceLast() << " seconds." << std::endl;
    }

    return CollOfScalar(u.value());
}

//CollOfScalar EquelleRuntimeCUDA::solveForUpdate(const CollOfScalar& residual) const {
//    return residual;
//}


#endif // EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED
