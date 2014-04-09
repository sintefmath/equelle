#include <opm/core/grid/GridManager.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

#include <iostream>

#include "EquelleRuntimeCUDA.hpp"
#include "CudaMatrix.hpp"

using namespace equelleCUDA;

/* 
   This test file is ment to test the class CudaMatrix!

   It relays on the cusparse library which needs a global 
   variable created by the EquelleRuntimeCUDA constructor.
   Therefore, we need to read the grid.
   
*/


int main(int argc, char** argv) {

    if (argc < 2) {
	std::cout << "Need a parameter file, please!\n";
	return 1;
    }

    Opm::parameter::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);

    std::cout << "Creating an empty matrix:\n";
    CudaMatrix A;
    std::cout << "Success! :) \n";
    
    std::cout << "Nnz: " << A.getNnz() << std::endl;

    return 0;
}
