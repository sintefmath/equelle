

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <iostream>
#include <exception>
#include <string>

#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "equelleTypedefs.hpp"
#include "equelleTypedefs.hpp"

#include "gridTest.h"


using namespace equelleCUDA;

// Function for comparing a collection to a known hard coded solution
template <int dummy>
int compare_collection(CollOfIndices<dummy> coll, int sol[], 
		       int sol_size, bool full,
		       std::string test) 
{ 
    // Testing full:
    if ( coll.isFull() != full ) {
	std::cout << "Error in gridTest.cu - testing " << test << "\n";
	if ( full ) {
	    std::cout << "\tCollection should be full, but is not.\n";
	    return 1;
	}
	else {
	    std::cout << "\tCollection should not be full, but is.\n";
	    return 1;
	}
    }
    if ( coll.isFull() ) {
	// Nothing more to test
	return 0;
    }
    
    // Test size:
    if ( coll.size() != sol_size ) {
	std::cout << "Error in gridTest.cu - testing " << test << "\n";
	std::cout << "\tThe collection is of wrong size!\n";
	std::cout << "\tSize is " << coll.size() << " but should be " << sol_size << "\n";
	return 1;
    }


    // Testing indices
    thrust::host_vector<int> host = coll.toHost();
    std::cout << "Collection " << test << " is the following:\n";
    bool correct = true;
    for (int i = 0; i < host.size(); ++i) {
	std::cout << host[i] << " ";
	if (i < sol_size) {
	    if (host[i] != sol[i]) {
		std::cout << "(<- " << sol[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in gridTest.cu - testing " << test << "\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	return 1;
    }
    
    return 0;

}


int compare_collection_bool(CollOfBool coll, bool sol[], 
		       int sol_size, std::string test) 
{ 
    // Testing indices
    thrust::host_vector<bool> host = coll;
    std::cout << "Collection of Booleans " << test << " is the following:\n";
    bool correct = true;
    for (int i = 0; i < host.size(); ++i) {
	std::cout << host[i] << " ";
	if (i < sol_size) {
	    if (host[i] != sol[i]) {
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in gridTest.cu - testing " << test << "\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	return 1;
    }

    if ( coll.size() != sol_size ) {
	std::cout << "Error in gridTest.cu - testing " << test << "\n";
	std::cout << "\tThe collection is of wrong size!\n";
	std::cout << "\tSize is " << coll.size() << " but should be " << sol_size << "\n";
	return 1;
    }
    
    return 0;

}


// Test functions return 0 for success, 1 otherwise

// This test expect 4x3 grid


int cuda_main(DeviceGrid dg) {
    
    std::cout << "From cuda_main!\n";

    DeviceGrid dg2(dg);


    std::cout << "Test:  (4?) " << dg.setID(1)  << std::endl;
    std::cout << "Test2: (4?) " << dg2.setID(2) << std::endl;

    // Test allCells:
    int empty_array[] = {};
    if ( compare_collection(dg.allCells(), empty_array, 0, true, "allCells()") ) {
	return 1;
    }

    // Test allFaces:
    if ( compare_collection(dg.allFaces(), empty_array, 0, true, "allFaces()") ) {
	return 1;
    }
    
    // Test boundaryFaces()
    int boundary_faces[] = {0, 4, 5, 9, 10, 14, 15, 16, 17, 18, 27, 28, 29, 30 };
    if ( compare_collection(dg.boundaryFaces(), boundary_faces, 14, false, "boundaryFaces()") ) {
	return 1;
    }

    // Test interiorFaces()
    int int_faces[] = {1,2,3,6,7,8,11,12,13,19,20,21,22,23,24,25,26};
    if ( compare_collection(dg.interiorFaces(), int_faces, 17, false, "interiorFaces()") ) {
	return 1;
    }

    // Test boundaryCells()
    int bnd_cells[] =  {0,1,2,3,4,7,8,9,10,11};
    if ( compare_collection(dg.boundaryCells(), bnd_cells, 10, false, "boundaryCells()") ) {
	return 1;
    }

    // Test interiorCells()
    int int_cells[] = {5,6};
    if ( compare_collection(dg.interiorCells(), int_cells, 2, false, "interiorCells()") ) {
	return 1;
    }
    
    // Test firstCell(allFaces())
    int first_cells[] = {-1,0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11};
    if ( compare_collection(dg.firstCell(dg.allFaces()), first_cells, 31, false, 
			    "firstCell(allFaces())") ) {
	return 1;
    }
    
    // Test secondCell(allFaces())
    int second_cells[] = {0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,0,1,2,3,4,5,6,7,8,9,10,11,-1,-1,-1,-1};
    if ( compare_collection(dg.secondCell(dg.allFaces()), second_cells, 31, false, 
			    "secondCell(allFaces())") ) {
	return 1;
    }

    // Test firstCell(interiorFaces())
    int first_int[] = {0,1,2,4,5,6,8,9,10,0,1,2,3,4,5,6,7};
    if ( compare_collection(dg.firstCell(dg.interiorFaces()), first_int, 17, false,
			    "firstCell(interiorFaces())") ) {
	return 1;
    }

    // Test secondCell(interiorFaces())
    int second_int[] = {1,2,3,5,6,7,9,10,11,4,5,6,7,8,9,10,11};
    if ( compare_collection(dg.secondCell(dg.interiorFaces()), second_int, 17, false,
			    "secondCell(interiorFaces())") ) {
	return 1;
    }

    // Test firstCell(boundaryFaces())
    int first_bound[] = {-1,3,-1,7,-1,11,-1,-1,-1,-1,8,9,10,11};
    if ( compare_collection(dg.firstCell(dg.boundaryFaces()), first_bound, 14, false,
			    "firstCell(boundaryFaces())") ) {
	return 1;
    }
    
    // Test secondCell(boundaryFaces())
    int second_bound[] = {0,-1,4,-1,8,-1,0,1,2,3,-1,-1,-1,-1};
    CollOfCell second_bound_cells = dg.secondCell(dg.boundaryFaces());
    if ( compare_collection(second_bound_cells, second_bound, 14, false,
			    "secondCell(boundaryFaces())") ) {
	return 1;
    }

    // Testing isEmpty:
    bool second_bound_empty[] = {0, 1,0,1,0,1,0,0,0,0,1,1,1,1};
    if ( compare_collection_bool(second_bound_cells.isEmpty(), 
				 second_bound_empty, 14,
				 "isEmpty(secondCell(boundaryFaces()))")) {
	return 1;
    }


    // THIS GIVES AN COMPILER ERROR - AND IT SHOULD :)
    // Test firstCell(interiorCells()
    //if ( compare_collection( dg.firstCell(dg.interiorCells()), {}, 0, false,
    //			     "firstCell(interiorCells())") ) {
    //	return 1;
    //}
							    

    return 0;
}

