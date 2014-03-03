
#include <iostream>

#include <thrust/host_vector.h>
#include <vector>

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"


using namespace equelleCUDA;



int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test);



//test_suite* init_unit_test_suite( int argc, char* argv[] )
//{
int main( int argc, char** argv) {
    
    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);
    
    DeviceGrid dg(er.getGrid());
    
    CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());
    double a_full_sol[] = {0,10,20,30,40,50,60,70,80,90,100,110};
    int a_full_size = 12;
    if ( compare(a, a_full_sol, a_full_size, "inputCollectionOfScalar(a)") ) {
	return 1;
    }

    CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());
    double b_full_sol[] = {124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124};
    int b_full_size = 12;
    if ( compare(b, b_full_sol, b_full_size, "inputCollectionOfScalar(b)") ) {
	return 1;
    }

    CollOfScalar faces = er.inputCollectionOfScalar("faces", er.allFaces());
    double faces_sol[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    for (int i = 0; i < 31; i++) {
	faces_sol[i] *= i*10;
    }
    int faces_size = 31;
    if ( compare(faces, faces_sol, faces_size, "inputaCollectionOfScalar(faces)") ) {
	return 1;
    }

    return 0;
}


int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test) 
{ 
    // Test size:
    if ( scal.size() != sol_size ) {
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe collection is of wrong size!\n";
	std::cout << "\tSize is " << scal.size() << " but should be " << sol_size << "\n";
	return 1;
    }
    
    // Testing indices
    std::vector<double> host = scal.copyToHost();
    std::cout << "CollOfScalar " << test << " is the following:\n";
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
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	return 1;
    }

    return 0;

}
