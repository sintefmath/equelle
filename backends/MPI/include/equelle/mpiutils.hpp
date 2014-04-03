#pragma once

#include <vector>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <Eigen/Sparse>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

struct UnstructuredGrid;

#define MPI_SAFE_CALL( err ) equelle::mpiSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )
#define ZOLTAN_SAFE_CALL( err ) equelle::zoltanSafeCall( err, __FILE__, __LINE__, __FUNCTION__ )

namespace equelle {

/**
 * @brief The MPIInitializer class is responsible for initializing and finalizing MPI and Zoltan.
 *
 * The MPIInitializer class is responsible for initializing and finalizing MPI and Zoltan.
 * It is intended to be used in program execution points (main(), BOOST_GLOBAL_FIXTURE etc.)
 * The behaviour of MPI is undefined if it is initialized or finalized more than once, so
 * the recommended practice is to use this class as a singleton, this is not enforced to allow
 * for easy integration with test frameworks.
 */
class MPIInitializer {
public:
    MPIInitializer();
    ~MPIInitializer();
};


/**
 * @brief zoltanSafeCall is intended to be used with the ZOLTAN_SAFE_CALL macro to wrap around all calls to Zoltan to test for errors and report them.
 * @param err
 * @param file
 * @param line
 * @param functionName
 */
void zoltanSafeCall( const int err, const std::string file, const int line, const std::string functionName );

/**
 * @brief mpiSafeCallis intended to be used with the MPI_SAFE_CALL macro to wrap around all calls to MPI to test for errors and report them.
 * @param err
 * @param file
 * @param line
 * @param functionName
 */
void mpiSafeCall( const int err, const std::string file, const int line, const std::string functionName );

int getMPIRank();

int getMPISize();

/**
 * @brief dumpEigenCSR is a debug function to make it easy to import Eigen matrix into other libraries.
 *
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
void dumpEigenCSR( const Eigen::SparseMatrix<double>& mat, std::ostream& s = std::cout  );

void dumpGrid( const UnstructuredGrid* grid );

template<class T>
void dumpVector( const std::vector<T>& v, std::ostream& s = std::cout ) {
    std::copy( begin( v ), end( v ), std::ostream_iterator<T>( s, " " ) );
    s << std::endl;
}

template<class T>
void injectMockData( Opm::parameter::ParameterGroup& param, std::string key, T begin, T end ) {
    std::string filename = key + ".mockdata";
    param.insertParameter( key + "_from_file", "true" );
    param.insertParameter( key + "_filename", filename );

    std::ofstream f(filename);
    std::copy( begin, end, std::ostream_iterator<typename T::value_type>( f, " " ) );
}

template<class T>
void dumpArray( const  std::vector<T>& v, std::ostream& s = std::cout  ) {
    std::copy( v.begin(), v.end(), std::ostream_iterator<T>( s, " " ) );
    s << std::endl;
}


} // namespace equelle

