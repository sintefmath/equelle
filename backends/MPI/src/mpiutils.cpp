#include "equelle/mpiutils.hpp"

namespace equelle {

MPIInitializer::MPIInitializer()
{
    MPI_SAFE_CALL( MPI_Init( NULL, NULL ) );

    int size;
    MPI_SAFE_CALL( MPI_Comm_size( MPI_COMM_WORLD, &size ) );

    float zoltanVersion;
    ZOLTAN_SAFE_CALL( Zoltan_Initialize( 0, NULL, &zoltanVersion ) );
}

MPIInitializer::~MPIInitializer()
{
     MPI_SAFE_CALL( MPI_Finalize() );
}

void zoltanSafeCall(const int err, const std::string file, const int line, const std::string functionName) {
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

void mpiSafeCall(const int err, const std::string file, const int line, const std::string functionName) {
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

int getMPIRank() {
    int rank;
    MPI_SAFE_CALL( MPI_Comm_rank( MPI_COMM_WORLD, &rank ) );

    return rank;
}

void dumpEigenCSR(const Eigen::SparseMatrix<double> &mat, std::ostream &s) {
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

void dumpGrid(const UnstructuredGrid *grid) {
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

    /*
    for( int i = 0; i < grid->number_of_faces; ++i ) {
        face_cells << i << ": [" << grid->face_cells[2*i] << ", " << grid->face_cells[2*i + 1 ] << "], ";
    }

    std::cerr << centroids.str() << std::endl;
    std::cerr << face_cells.str();
*/
    std::cout << "cell_facepos: ";
    std::copy_n( grid->cell_facepos, grid->number_of_cells + 1, std::ostream_iterator<int>( std::cout, " " ) );
}

} // namespace equelle

