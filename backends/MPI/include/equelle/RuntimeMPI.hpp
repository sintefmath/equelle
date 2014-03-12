#pragma once

#include <memory>
#include "equelle/mpiutils.hpp"
#include "equelle/ZoltanGrid.hpp"

class Zoltan;
namespace Opm {
    class GridManager;
}

namespace equelle {


/** RuntimeMPI is responsible for executing Equelle-simulators using MPI.
 *  It handles both the MPI context and the domain decomposition, using Zoltan. */
class  RuntimeMPI {
public:
    RuntimeMPI();
    virtual ~RuntimeMPI();

    std::unique_ptr<Zoltan> zoltan;
    std::unique_ptr<Opm::GridManager> grid_manager;

    equelle::zoltanReturns computePartition();
    void initializeZoltan();
    void initializeGrid();        
};

}
