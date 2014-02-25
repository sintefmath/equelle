

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "deviceGrid.hpp"

#include "gridTest.h"


using namespace equelleCUDA;


int cuda_main(DeviceGrid dg) {
    
    std::cout << "From cuda_main!\n";

    DeviceGrid dg2(dg);


    std::cout << "Test:  (4?) " << dg.test(10)  << std::endl;
    std::cout << "Test2: (4?) " << dg2.test(20) << std::endl;

    

    return 0;
}



// more to come!
