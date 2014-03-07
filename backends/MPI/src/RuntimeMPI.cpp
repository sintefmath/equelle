#include "equelle/RuntimeMPI.hpp"
#include <mutex>
#include <iostream>

#include <mpi.h>
//#include <zoltan_cpp.h>

#include "equelle/mpiutils.hpp"


namespace equelle {

namespace impl {
void initMPIandZoltan() {
    MPI_SAFE_CALL( MPI_Init( NULL, NULL ) );

    int size;
    MPI_SAFE_CALL( MPI_Comm_size( MPI_COMM_WORLD, &size ) );

    std::cerr << "Size: " << size << std::endl;
    //float zoltanVersion;
    //ZOLTAN_SAFE_CALL( Zoltan_Initialize( argc, argv, &zoltanVersion ) );

    int rank;
    MPI_SAFE_CALL( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) );

    //std::cerr << "Hello from: " << rank << std::endl;
}

}

std::once_flag flag;

RuntimeMPI::RuntimeMPI()
{

    std::string progname("a.out");
    //int argc = 1;
    //char* argv[] = { strdup( progname.c_str() ) };
    impl::initMPIandZoltan();
    //std::call_once( flag, impl::initMPIandZoltan );



    
}

RuntimeMPI::~RuntimeMPI()
{
    MPI_SAFE_CALL( MPI_Finalize() );
}


} // namespace equlle
