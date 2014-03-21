#pragma once

#include <memory>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

#include "equelle/equelleTypes.hpp"
#include "equelle/mpiutils.hpp"
#include "equelle/ZoltanGrid.hpp"
#include "equelle/SubGridBuilder.hpp"

class Zoltan;
namespace Opm {
class GridManager;
}

namespace equelle {
class EquelleRuntimeCPU;

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
    RuntimeMPI( const Opm::parameter::ParameterGroup& param );
    virtual ~RuntimeMPI();

    std::unique_ptr<Opm::GridManager> globalGrid; //! Assumed to be read from disk on every node.
    equelle::SubGrid subGrid; //! Filled with the local subGrid after call to decompose.

    void decompose();
    equelle::zoltanReturns computePartition();

    ///@{ Topology and geometry related.
    CollOfCell allCells() const;
    ///@}

    /// Return the number of cells in collection. Will do MPI-transfer.
    int globalCollectionSize( const CollOfFace& coll );

    ///@{ Input
    CollOfScalar inputCollectionOfScalar(const String& name,
                                         const CollOfFace& coll);

    CollOfScalar inputCollectionOfScalar(const String& name,
                                         const CollOfCell& coll);
    ///@}

private:
    std::unique_ptr<Zoltan> zoltan;
    std::unique_ptr<equelle::EquelleRuntimeCPU> runtime;
    Opm::parameter::ParameterGroup param_;


    void initializeZoltan();
    void initializeGrid();

};

} // namespace equelle

#include "RuntimeMPI_impl.hpp"

