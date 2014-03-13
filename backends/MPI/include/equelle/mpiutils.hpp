#pragma once

#include <stdexcept>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <mpi.h>

#include <zoltan_cpp.h>

#include <Eigen/Eigen>
#include <Eigen/Sparse>

#include "opm/core/grid.h"

#define MPI_SAFE_CALL( err ) equelle::mpiSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )
#define ZOLTAN_SAFE_CALL( err ) equelle::zoltanSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )

namespace equelle {


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

inline
int getMPIRank() {
    int rank;
    MPI_SAFE_CALL( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) );

    return rank;
}


/**
 * @brief dumpEigenCSR is a debug function to make it easy to import Eigen matrix into other libraries.
 * The following python code exemplifies how it can be imported into Python:
 * f = open('csr.txt')
 * lines = f.readlines()
 * f.close()
 * inner = numpy.array( lines[0].strip().split() ).astype(int)
 * outer = numpy.array( lines[1].strip().split() ).astype(int)
 * data = numpy.array( lines[2].strip().split() ).astype(float)
 * scipy.sparse.csc_matrix( (data,inner, outer) ).todense()
 * @param mat Matrix to dump
 * @param s stream/file to dump to
 */
inline
void dumpEigenCSR( const Eigen::SparseMatrix<double>& mat, std::ostream& s = std::cout  ) {
    Eigen::SparseMatrix<double> comp = mat;

    comp.makeCompressed();

    std::copy_n( comp.innerIndexPtr(), comp.nonZeros(),
                 std::ostream_iterator<int>( s, " " ) );
    s << std::endl;

    std::copy_n( comp.outerIndexPtr(), comp.outerSize()+1,
                 std::ostream_iterator<int>( s, " " ) );
    s << std::endl;

    std::copy_n( comp.valuePtr(), comp.nonZeros(),
                 std::ostream_iterator<double>( s, " " ) );
    s << std::endl;
}

inline
void dumpGrid( const UnstructuredGrid* grid ) {
    std::stringstream centroids;
    std::stringstream face_cells;
    std::stringstream global_cell;
    const auto dim = grid->dimensions;

    centroids << "Centroids: ";
    face_cells << "Face cells: ";
    global_cell << "global_cell: ";

    for( int i = 0; i < grid->number_of_cells; ++i ) {
        centroids << "[";
        std::copy( &grid->cell_centroids[i*dim], &grid->cell_centroids[i*dim + dim],
                   std::ostream_iterator<double>( centroids, " " ) );
        centroids << "]";

        //global_cell << grid->global_cell[i] << " ";
    }


    for( int i = 0; i < grid->number_of_faces; ++i ) {
        face_cells << i << ": [" << grid->face_cells[2*i] << ", " << grid->face_cells[2*i + 1 ] << "], ";
    }

    std::cerr << centroids.str() << std::endl;
    std::cerr << face_cells.str();
}




} // namespace equelle

