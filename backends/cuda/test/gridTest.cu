

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <iostream>
#include <exception>
#include <string>

#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"


#include "gridTest.h"


using namespace equelleCUDA;

// Function for comparing a collection to a known hard coded solution
int compare_collection(CollOfIndices coll, int sol[], 
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
    
    // Testing indices
    thrust::host_vector<int> host = coll.toHost();
    std::cout << "Collection " << test << " is the following:\n";
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
    
    // Test firstCell()
    int first_cells[] = {-1,0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11};
    if ( compare_collection(dg.firstCell(), first_cells, 31, false, "firstCell()") ) {
	return 1;
    }
    
    // Test secondCell()
    int second_cells[] = {0,1,2,3,-1,4,5,6,7,-1,8,9,10,11,-1,0,1,2,3,4,5,6,7,8,9,10,11,-1,-1,-1,-1};
    if ( compare_collection(dg.secondCell(), second_cells, 31, false, "secondCell()") ) {
	return 1;
    }

    return 0;
}

