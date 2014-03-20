#include <iostream>
#include <fstream>

#include <mpi.h>

#include "equelle/mpiutils.hpp"
#include "equelle/RuntimeMPI.hpp"
#include "opm/core/grid.h"
#include "opm/core/grid/GridManager.hpp"


int main( int argc, char* argv[] ) {
    MPI_SAFE_CALL( MPI_Init( NULL, NULL ) );

    int size;
    MPI_SAFE_CALL( MPI_Comm_size( MPI_COMM_WORLD, &size ) );

    if ( argc != 2 ) {
        std::cerr << "Usage: " << argv[0] << " " << "gridfile\n" << std::endl;
    } else {
        equelle::RuntimeMPI runtime;
        if ( equelle::getMPIRank() == 0 ) {
            runtime.globalGrid.reset( new Opm::GridManager( argv[1] ) );
        }

        auto zr = runtime.computePartition();

        if ( equelle::getMPIRank() == 0 ) {
            std::ofstream f( std::string( argv[1]) + std::string( ".part.out")  );
            equelle::ZoltanGrid::dumpRank0Exports( runtime.globalGrid->c_grid()->number_of_cells, zr, f );
        }
    }

    MPI_SAFE_CALL( MPI_Finalize() );
    return 1;
}
