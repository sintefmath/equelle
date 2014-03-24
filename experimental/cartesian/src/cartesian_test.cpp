#define BOOST_TEST_MAIN
#define BOOST_TEST_MODULE EquelleCartesianTest

#include <memory>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iterator>
#include <vector>
#include <numeric>
#include <fstream>
#include <tuple>

#include <boost/test/unit_test.hpp>
#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/CartesianGrid.hpp"

namespace {
    template<class T>
    void injectMockData( Opm::parameter::ParameterGroup& param, std::string key, T begin, T end ) {
        std::string filename = key + ".mockdata";
        param.insertParameter( key + "_from_file", "true" );
        param.insertParameter( key + "_filename", filename );

        std::ofstream f(filename);
        std::copy( begin, end, std::ostream_iterator<typename T::value_type>( f, " " ) );
    }

}

BOOST_AUTO_TEST_CASE( generateCartesianGrid ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 1;

    Opm::parameter::ParameterGroup param;
    std::vector<double> default_waveheights( dim_x * dim_y, 1.0 );

    //injectMockData( param, "waveheights", default_waveheights.begin(), default_waveheights.end() );

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );

    BOOST_CHECK_EQUAL( grid.cartdims[0], dim_x );
    BOOST_CHECK_EQUAL( grid.cartdims[1], dim_y );
    BOOST_CHECK_EQUAL( grid.cartdims[2], -1 );
    BOOST_CHECK_EQUAL( grid.dimensions, 2 );

    BOOST_CHECK_EQUAL( grid.number_of_cells, dim_x*dim_y );

    BOOST_CHECK_EQUAL( grid.number_of_cells_and_ghost_cells, (dim_x+2*ghostWidth)*(dim_y+2*ghostWidth) );

    // Collection of scalar with number of elements = (dim_x + 2*ghost) * (dim_y + 2*ghost)
    equelle::CartesianGrid::CartesianCollectionOfScalar waveheights_0 = grid.inputCellScalarWithDefault( "waveheights", 1.0 );
    BOOST_REQUIRE_EQUAL( waveheights_0.size(), grid.number_of_cells_and_ghost_cells );
    BOOST_CHECK_EQUAL( *grid.cellAt( 0, 0, waveheights_0 ), 1.0 );
    BOOST_CHECK_EQUAL( *grid.cellAt( -1, -1, waveheights_0 ), 0.0 );


    equelle::CartesianGrid::CartesianCollectionOfScalar waveheights_1( waveheights_0.size() );

    const double* waveheights_0_ptr = grid.cellAt( 0, 0, waveheights_0 );
    double* waveheights_1_ptr = grid.cellAt( 0, 0, waveheights_1 );

    int stride_x = grid.getStride( equelle::Dimension::x );
    BOOST_REQUIRE_EQUAL( stride_x, dim_x + 2*ghostWidth );

    int stride_y = grid.getStride( equelle::Dimension::y );

    const double k = 1.0/8.0;

    for( int j = 0; j < grid.cartdims[1]; ++j ) {
        for( int i = 0; i < grid.cartdims[0]; ++i ) {
            waveheights_1_ptr[j*stride_x + i*stride_y] =    k * ( waveheights_0_ptr[(j-1)*stride_x + (i-1)*stride_y] +
                                                                  waveheights_0_ptr[(j-1)*stride_x + (i+1)*stride_y] +
                                                                  waveheights_0_ptr[(j+1)*stride_x + (i-1)*stride_y] +
                                                                  waveheights_0_ptr[(j+1)*stride_x + (i+1)*stride_y] +
                                                              4.0*waveheights_0_ptr[(j-0)*stride_x + (i-0)*stride_y] );
        }
    }
}
