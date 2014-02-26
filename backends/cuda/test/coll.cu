#include <iostream>

#include "deviceGrid.hpp"
#include "CollOfIndices.hpp"


#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>

#include <opm/core/utility/ErrorMacros.hpp>

using namespace equelleCUDA;


void test_copy(CollOfIndices coll) {
    if ( coll.size() != 10 ) {
	OPM_THROW(std::runtime_error, "\ntest_copy - size don't match");
    }
    if ( coll.isFull() ) {
	OPM_THROW(std::runtime_error, "\ntest_copy - collection says it is " << coll.isFull() << " but should be false.");
    }
    for (int i = 0; i < coll.size(); i++) {
	if ( coll[i] != 2 ){
	    OPM_THROW(std::runtime_error, "\ntest_copy - Collection don't match. Expected 2, got " << coll[i]);
	}
    }
}

void test_full(CollOfIndices coll) {
    if ( !coll.isFull() ) {
	OPM_THROW(std::runtime_error, "\ntest_full - isFull() should be true but is " << coll.isFull());
    }
    if ( coll.size() != 0 ) {
	OPM_THROW(std::runtime_error, "\ntest_full - vector should be empty, but has size " << coll.size());
    }

}

void test_back_to_host(CollOfIndices coll) {
    if (coll.size() != 20) {
	OPM_THROW(std::runtime_error, "\ntest_back_to_host - size should be 20, is " << coll.size());
    }
    thrust::host_vector<int> back = coll.toHost();
    for(int i = 0; i < back.size(); i++) {
	if (back[i] != i) {
	    OPM_THROW(std::runtime_error, "\ntest_back_to_host - back[" << i << "] should be " << i << " but is " << back[i]);
	}
    }


}

int cuda_main() {

    thrust::host_vector<int> host(10, 2);
    CollOfIndices coll(host);

    test_copy(coll);
    
    CollOfIndices coll2 = coll;
    test_copy(coll2);

    CollOfIndices coll3(true);
    test_full(coll3);

    thrust::host_vector<int> host2(0);
    for( int i = 0; i < 20; i++) {
	host2.push_back(i);
    }
    CollOfIndices coll4(host2);
    test_back_to_host(coll4);

    return 0;
}