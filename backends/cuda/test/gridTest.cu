

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <iostream>
#include <exception>


#include "deviceGrid.hpp"

#include "gridTest.h"


using namespace equelleCUDA;

// Test functions return 0 for success, 1 otherwise

// This test expect 4x3 grid

int all_cells_test(DeviceGrid dg) {
    //Expect the Collection to be full.

    Collection coll = dg.allCells();
    if ( !coll.isFull() ) {
	std::cout << "Error in gridTest.cu - all_cells_test\n";
	return 1;
    }
    std::cout << "Passed all_cells_test\n";
    return 0;

}

int all_faces_test(DeviceGrid dg) {
    // Expect the Collection to be full
    Collection coll = dg.allFaces();
    if ( !coll.isFull() ) {
	std::cout << " Error in gridTest.cu - all_faces_test\n";
	return 1;
    }
    std::cout << "Passed all_faces_test\n";
    return 0;
}

int boundary_faces_test(DeviceGrid dg) {
    // Expecting a non-full collection containing
    // { 0 4 5 9 10 14 15 16 17 18 27 28 29 30 }
    int lf[] = {0, 4, 5, 9, 10, 14, 15, 16, 17, 18, 27, 28, 29, 30 };
    int lf_size = 14;
    Collection coll = dg.boundaryFaces();
    if ( coll.isFull() ) {
	std::cout << "Error in gridTest.cu - boundary_faces_test\n";
	std::cout << "\tCollection should not be full\n";
	return 1;
    }

    thrust::host_vector<int> host = coll.toHost();
    std::cout << "Boundary faces is the following:\n";
    bool correct = true;
    for (int i = 0; i < host.size(); ++i) {
	std::cout << host[i] << " ";
	if (i < lf_size) {
	    if (host[i] != lf[i]) {
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in gridTest.cu - boundary_faces_test\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	return 1;
    }

    std::cout << "Testing size\n";
    if ( coll.size() != lf_size ) {
	std::cout << "Error in gridTest.cu - boundary_faces_test\n";
	std::cout << "\tThe collection is of wrong size!\n";
    }
    
    return 0;
}

int cuda_main(DeviceGrid dg) {
    
    std::cout << "From cuda_main!\n";

    DeviceGrid dg2(dg);


    std::cout << "Test:  (4?) " << dg.setID(1)  << std::endl;
    std::cout << "Test2: (4?) " << dg2.setID(2) << std::endl;

    if ( all_cells_test(dg) ) {
	return 1;
    }
    if ( all_faces_test(dg) ) {
	return 1;
    }
    if ( boundary_faces_test(dg) ) {
	return 1;
    }

    return 0;
}



// more to come!
