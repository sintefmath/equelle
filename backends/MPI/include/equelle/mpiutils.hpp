#pragma once

#include <stdexcept>
#include <sstream>
#include <mpi.h>

//#include <zoltan_cpp.h>


#define MPI_SAFE_CALL( err ) equelle::mpi::mpiSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )
#define ZOLTAN_SAFE_CALL( err ) equelle::mpi::zoltanSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )

namespace equelle { namespace mpi {

/*
inline
void zoltanSafeCall( const int err, const std::string file, const int line, const std::string functionName ) {
    if ( err != ZOLTAN_OK ) {
        std::stringstream ss;

        ss << "Zoltan returned: ";
        switch( err ) {
        case ZOLTAN_WARN:
            ss << "ZOLTAN_WARN";
            break;
        case ZOLTAN_FATAL:
            ss << "ZOLTAN_FATAL";
            break;
        case ZOLTAN_MEMERR:
            ss << "ZOLTAN_MEMERR";
            break;
        default:
            ss << "Unknown Zoltan error code";
        }

        ss << "File: " << file << std::endl;
        ss << "Line: " << line << std::endl;
        ss << "Function" << functionName << std::endl;
        throw std::runtime_error( ss.str() );
    }
}
*/
inline
void mpiSafeCall( const int err, const std::string file, const int line, const std::string functionName ) {
    if ( err == MPI_SUCCESS ) {
        return;
    }

    int errlen = 0;
    char message[MPI_MAX_ERROR_STRING]{};
    MPI_Error_string( err, message, &errlen );

    std::stringstream ss;
    ss << "MPI error code: " << err << " " << message;
    ss << "at File: " << file << std::endl;
    ss << "Line: " << line << std::endl;
    ss << "Function: " << functionName << std::endl;

    throw std::runtime_error( ss.str() );
}




}} // namespace equelle::mpi

