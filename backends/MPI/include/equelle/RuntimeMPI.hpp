#pragma once

#include <memory>
#include "equelle/mpiutils.hpp"
#include "equelle/ZoltanGrid.hpp"
#include "equelle/SubGridBuilder.hpp"


class Zoltan;
namespace Opm {
    class GridManager;
}

namespace equelle {


/** RuntimeMPI is responsible for executing Equelle-simulators using MPI.
 *  It handles both the MPI context and the domain decomposition, using Zoltan.
 *
 *  The current domain-decomposition and subgrid-building relies on all nodes
 *  reading the globalGrid from disk and then extracting the local subGrid from it.
 *  This is because the current subGrid-building (with ghost cells) relies on full
 *  access to the neighborhood.
 */
class  RuntimeMPI {
public:
    RuntimeMPI();
    virtual ~RuntimeMPI();

    std::unique_ptr<Opm::GridManager> globalGrid; //! Assumed to be read from disk on every node.
    equelle::SubGrid subGrid; //! Filled with the local subGrid after call to decompose.

    void decompose();

    equelle::zoltanReturns computePartition();
private:
    std::unique_ptr<Zoltan> zoltan;
    void initializeZoltan();
    void initializeGrid();        

};

}
