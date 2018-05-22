// This file provides unit tests for the class
// DeviceGrid, which handles grid related functionality on the GPU.

//#include <boost/test/included/unit_test.hpp>
//using namespace boost::unit_test;

// CUDA code and BOOST don't mix. Need to write tests without the BOOST framework.
//
// This file tests grid relations
//    all, interior, boundary, first/second and trinary if. 
//    on a 3*4 grid.


#include <cuda.h>
#include <cuda_runtime.h>

#include "gridTest.h"

// Include everything that the equelle runtime need
#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/grid/UnstructuredGrid.h>
#include <opm/grid/GridManager.hpp>

#include <opm/common/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"

using namespace equelleCUDA;


// Test for run-time environment:
int runtime_test(const EquelleRuntimeCUDA& er);


void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}


void equal_test_function(int a, int b) {
    if ( a != b ) {
	OPM_THROW(std::runtime_error, "\nequal_test_function - " << a << " != " << b);
    }
}


int main( int argc, char** argv )
{
    int ret = 1;
    try {
	// Get user parameters
	Opm::ParameterGroup param( argc, argv, false);

	// Create the Equelle runtime
	EquelleRuntimeCUDA er(param);
	ensureRequirements(er);
	
	// Get the device grid so that we can play around with it!
	DeviceGrid dg(er.getGrid());
    
	int a = 1;
	int b = 1;
	//framework::master_test_suite().add( BOOST_REQUIRE( a == b ), 0);
	
	equal_test_function(a, b);
	

	//int out = cuda_main(dg);
	//std::cout << "Back in main!\n";
	//return out;
	ret = cuda_main(dg);

	if ( ret == 0 ) {
	    std::cout << "Testing from Runtime!\n";
	    ret = runtime_test(er);
	}
    }
    catch (...) {
	std::cerr << "\n\n FOUND EXCEPTION!\n\n\n";
	ret = 1;
    }
    return ret;
}




int runtime_test( const EquelleRuntimeCUDA& er) {

    CollOfCell allC = er.allCells();
    int empty_sol[] = {};
    int empty_size = 0;
    if ( compare_collection(allC, empty_sol, empty_size, true, 
			    "Runtime.allCells();") )  {
	return 1;
    }
   // Test allCells:

    // Test allFaces:
    if ( compare_collection(er.allFaces(), empty_sol, empty_size, true,
			    "Runtime.allFaces()") ) {
	return 1;
    }
    
    // Test boundaryFaces()
    int boundary_faces[] = {0, 4, 5, 9, 10, 14, 15, 16, 17, 18, 27, 28, 29, 30 };
    if ( compare_collection(er.boundaryFaces(), boundary_faces, 14, false, 
			    "Runtime.boundaryFaces()") ) {
	return 1;
    }

    // Test interiorFaces()
    int int_faces[] = {1,2,3,6,7,8,11,12,13,19,20,21,22,23,24,25,26};
    if ( compare_collection(er.interiorFaces(), int_faces, 17, false, 
			    "Runtime.interiorFaces()") ) {
	return 1;
    }

    // Test boundaryCells()
    int bnd_cells[] =  {0,1,2,3,4,7,8,9,10,11};
    if ( compare_collection(er.boundaryCells(), bnd_cells, 10, false, 
			    "Runtime.boundaryCells()") ) {
	return 1;
    }

    // Test interiorCells()
    int int_cells[] = {5,6};
    if ( compare_collection(er.interiorCells(), int_cells, 2, false, 
			    "Runtime.interiorCells()") ) {
	return 1;
    }
    
    
    // Test firstCell(allFaces())
    int first_cells[] = {-1,0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11};
    if ( compare_collection(er.firstCell(er.allFaces()), first_cells, 31, false, 
			    "Runtime.firstCell(allFaces())") ) {
	return 1;
    }
    
    // Test secondCell(allFaces())
    int second_cells[] = {0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,0,1,2,3,4,5,6,7,8,9,10,11,-1,-1,-1,-1};
    if ( compare_collection(er.secondCell(er.allFaces()), second_cells, 31, false, 
			    "Runtime.secondCell(allFaces())") ) {
	return 1;
    }
    
    // Test firstCell(boundaryFaces())
    int first_bound_sol[] = {-1,3,-1,7,-1,11,-1,-1,-1,-1,8,9,10,11};
    CollOfCell first_bound = er.firstCell(er.boundaryFaces());
    if ( compare_collection(first_bound, first_bound_sol, 14, false,
			    "Runtime.firstCell(boundaryFaces())") ) {
	return 1;
    }
    
    // Test secondCell(boundaryFaces())
    int second_bound_sol[] = {0,-1,4,-1,8,-1,0,1,2,3,-1,-1,-1,-1};
    CollOfCell second_bound = er.secondCell(er.boundaryFaces());
    if ( compare_collection(second_bound, second_bound_sol, 14, false,
			    "Runtime.secondCell(boundaryFaces())") ) {
	return 1;
    }

    // Test for trinaryIf
    CollOfCell inner_bnd = er.trinaryIf(er.isEmpty(second_bound), first_bound, second_bound);
    int inner_bnd_sol[] = {0,3,4,7,8,11,0,1,2,3,8,9,10,11};
    if ( compare_collection(inner_bnd, inner_bnd_sol, 14, false,
			    "Runtime.trinaryIf(isEmpty(second), first, second)") ) {
	return 1;
    }
    
    return 0;

}
