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

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );

    BOOST_CHECK_EQUAL( grid.cartdims[0], dim_x );
    BOOST_CHECK_EQUAL( grid.cartdims[1], dim_y );    
    BOOST_CHECK_EQUAL( grid.dimensions, 2 );

    BOOST_CHECK_EQUAL( grid.number_of_cells, dim_x*dim_y );

    BOOST_CHECK_EQUAL( grid.number_of_cells_and_ghost_cells, (dim_x+2*ghostWidth)*(dim_y+2*ghostWidth) );

    // Collection of scalar with number of elements = (dim_x + 2*ghost) * (dim_y + 2*ghost)
    equelle::CartesianGrid::CartesianCollectionOfScalar waveheights_0 = grid.inputCellScalarWithDefault( "waveheights", 1.0 );


    BOOST_REQUIRE_EQUAL( waveheights_0.size(), grid.number_of_cells_and_ghost_cells );
    BOOST_CHECK_EQUAL( grid.cellAt( 0, 0, waveheights_0 ), 1.0 );
    BOOST_CHECK_EQUAL( grid.cellAt( -1, -1, waveheights_0 ), 0.0 );
    BOOST_CHECK_EQUAL( grid.cellAt( dim_x -1, dim_y -1, waveheights_0 ),  1.0 );

    equelle::CartesianGrid::CartesianCollectionOfScalar waveheights_1( waveheights_0.size() );



    int stride_x = grid.getStride( equelle::Dimension::x );
    BOOST_REQUIRE_EQUAL( stride_x, 1 );

    int stride_y = grid.getStride( equelle::Dimension::y );
    BOOST_REQUIRE_EQUAL( stride_y, dim_x + 2*ghostWidth );

    const double k = 1.0/8.0;

    for( int j = 0; j < dim_y; ++j ) {
        for( int i = 0; i < dim_x; ++i ) {
            grid.cellAt( i, j, waveheights_1 ) = k * ( grid.cellAt( i+0, j-1, waveheights_0 ) +
                                                       grid.cellAt( i+0, j+1, waveheights_0 ) +
                                                       grid.cellAt( i-1, j+0, waveheights_0 ) +
                                                       grid.cellAt( i+1, j+0, waveheights_0 ) +
                                                   4.0*grid.cellAt( i+0, j+0, waveheights_0 ) );
        }
    }

    std::ofstream f("waveheights_1.csv");
    grid.dumpGrid( waveheights_1, f );
}

BOOST_AUTO_TEST_CASE( faceTest ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 1;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );
/*
    equelle::CartesianGrid::CartesianCollectionOfScalar flux = grid.inputFaceScalarWithDefault( "permability", 0.5 );
    int i = 1; int j = 2;
    BOOST_CHECK_EQUAL( grid.faceAt( i, j, equelle::CartesianGrid::Face::negX, permability_0 ),
                       grid.faceAt( i-1, j, equelle::CartesianGrid::Face::posX, permability_0 ) );
*/
}
