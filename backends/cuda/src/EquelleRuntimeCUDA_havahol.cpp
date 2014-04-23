#include "EquelleRuntimeCUDA.hpp"
#include "CollOfScalar.hpp"
#include "CollOfIndices.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"
#include "wrapEquelleRuntime.hpp"

#include <string>
#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>


using namespace equelleCUDA;
using namespace wrapEquelleRuntimeCUDA;

//
//   CPP for implementation of non-cuda functions
//




void EquelleRuntimeCUDA::output(const String& tag, const CollOfScalar& coll)
{
    // Get data back to host
    std::vector<double> host = coll.copyToHost();
    
    if (output_to_file_) {
	// std::map<std::string, int> outputcount_;
	// set file name to tag-0000X.output
	int count = -1;
	auto it = outputcount_.find(tag);
	if ( it == outputcount_.end()) {
	    count = 0;
	    outputcount_[tag] = 1; // should contain the count to be used next time for same tag.
	} else {
	    count = outputcount_[tag];
	    ++outputcount_[tag];
	}
	std::ostringstream fname;
	fname << tag << "-" << std::setw(5) << std::setfill('0') << count << ".output";
	std::ofstream file(fname.str().c_str());
	if( !file ) {
	    OPM_THROW(std::runtime_error, "Failed to open " << fname.str());
	}
	file.precision(16);
	std::cout << "Printing to file...(host.size() = " << host.size() << " )\n";
	std::copy(host.data(), host.data() + host.size(),
		  std::ostream_iterator<double>(file, "\n"));
    } else {
	std::cout << "\n";
	std::cout << "Values in " << tag << std::endl;
	for(int i = 0; i < coll.size(); ++i) {
	    std::cout << host[i] << "  ";
	}
	std::cout << std::endl;
    }
}


Scalar EquelleRuntimeCUDA::inputScalarWithDefault(const String& name,
						  const Scalar default_value) {
    return param_.getDefault(name, default_value);
}



CollOfScalar EquelleRuntimeCUDA::trinaryIf( const CollOfBool& predicate,
					    const CollOfScalar& iftrue,
					    const CollOfScalar& iffalse) const {
    // First, we need same size of all input
    if (iftrue.size() != iffalse.size() || iftrue.size() != predicate.size()) {
	OPM_THROW(std::runtime_error, "Collections are not of the same size");
    }
    // Call a wrapper which calls a kernel
    return trinaryIfWrapper(predicate, iftrue, iffalse);
}


CollOfScalar EquelleRuntimeCUDA::gradient_old( const CollOfScalar& cell_scalarfield ) const {
    // This function is at the moment kept in order to be able to compare efficiency
    // against the new implementation, where we use the matrix from devOps_.

    // First, need cell_scalarfield to be defined on all cells:
    if ( cell_scalarfield.size() != dev_grid_.number_of_cells() ) {
	OPM_THROW(std::runtime_error, "Gradient need input defined on AllCells()");
    }
    
    return gradientWrapper(cell_scalarfield,
    			   dev_grid_.interiorFaces(),
    			   dev_grid_.face_cells(),
    			   devOps_.grad);
}

CollOfScalar EquelleRuntimeCUDA::gradient( const CollOfScalar& cell_scalarfield ) const {
    if ( cell_scalarfield.size() != dev_grid_.number_of_cells() ) {
	OPM_THROW(std::runtime_error, "Gradient need input defined on AllCells()");
    }

    if ( cell_scalarfield.useAutoDiff() ) {
	return CollOfScalar( devOps_.grad * cell_scalarfield.value(),
			     devOps_.grad * cell_scalarfield.derivative() );
    }
    // else: 
    return CollOfScalar( devOps_.grad * cell_scalarfield.value() );
}

CollOfScalar EquelleRuntimeCUDA::divergence_old(const CollOfScalar& face_fluxes) const {
    
    // If the size is not the same as the number of faces, then the input is
    // given as interiorFaces. Then it has to be extended to AllFaces.
    if ( face_fluxes.size() != dev_grid_.number_of_faces() ) {
	CollOfFace int_faces = interiorFaces();
	if ( face_fluxes.size() != devOps_.num_int_faces ) { // Then something wierd has happend
	    OPM_THROW(std::runtime_error, "Input for divergence has to be on AllFaces or on InteriorFaces.");
	}
	// Extend to AllFaces():
	CollOfScalar allFluxes = operatorExtend(face_fluxes, int_faces, allFaces());
	return divergenceWrapper(allFluxes,
				 dev_grid_,
				 devOps_.fulldiv);
    }
    else {
	// We are on allFaces already, so let's go!
	return divergenceWrapper(face_fluxes,
				 dev_grid_,
				 devOps_.fulldiv); 
    }
}

CollOfScalar EquelleRuntimeCUDA::divergence(const CollOfScalar& face_fluxes) const {
    
    // The input need to be defined on allFaces() or interiorFaces()
    if ( face_fluxes.size() != dev_grid_.number_of_faces() &&
	 face_fluxes.size() != devOps_.num_int_faces ) {
	OPM_THROW(std::runtime_error, "Input for divergence has to be on AllFaces or on InteriorFaces()");
    }
    
    if ( face_fluxes.size() == dev_grid_.number_of_faces() ) {
	if ( face_fluxes.useAutoDiff() ) {
	    return CollOfScalar( devOps_.fulldiv * face_fluxes.value(),
				 devOps_.fulldiv * face_fluxes.derivative() );
	}
	else {
	    return CollOfScalar( devOps_.fulldiv * face_fluxes.value() );
	}
    }
    else { // on internal faces
	if ( face_fluxes.useAutoDiff() ) {
	    return CollOfScalar( devOps_.div * face_fluxes.value(),
				 devOps_.div * face_fluxes.derivative() );
	}
	else {
	    return CollOfScalar( devOps_.div * face_fluxes.value() );
	}
    }
}


// SQRT
CollOfScalar EquelleRuntimeCUDA::sqrt(const CollOfScalar& x) const {

    return sqrtWrapper(x);
}



// Array Of {X} Collection Of Scalar
std::array<CollOfScalar, 1> equelleCUDA::makeArray( const CollOfScalar& t ) 
{
    return std::array<CollOfScalar, 1> {{t}};
}

std::array<CollOfScalar, 2> equelleCUDA::makeArray( const CollOfScalar& t1, 
						    const CollOfScalar& t2 )
{
    return std::array<CollOfScalar, 2> {{t1, t2}};
}

std::array<CollOfScalar, 3> equelleCUDA::makeArray( const CollOfScalar& t1,
						    const CollOfScalar& t2,
						    const CollOfScalar& t3 )
{
    return std::array<CollOfScalar, 3> {{t1, t2, t3}};
}

std::array<CollOfScalar, 4> equelleCUDA::makeArray( const CollOfScalar& t1,
						    const CollOfScalar& t2,
						    const CollOfScalar& t3,
						    const CollOfScalar& t4 )
{
    return std::array<CollOfScalar, 4> {{t1, t2, t3, t4}};
}
