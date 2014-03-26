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
#include <boost/format.hpp>
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

/**
 * Test construction of grid
 */
BOOST_AUTO_TEST_CASE( cartesianGridTest ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 2;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );

    BOOST_CHECK_EQUAL( grid.cartdims[0], dim_x );
    BOOST_CHECK_EQUAL( grid.cartdims[1], dim_y );    
    BOOST_CHECK_EQUAL( grid.dimensions, 2 );

    BOOST_CHECK_EQUAL( grid.number_of_cells, dim_x*dim_y );

    BOOST_CHECK_EQUAL( grid.number_of_cells_and_ghost_cells, (dim_x+2*ghostWidth)*(dim_y+2*ghostWidth) );

    int stride_x = grid.getStride( equelle::Dimension::x );
    BOOST_REQUIRE_EQUAL( stride_x, 1 );

    int stride_y = grid.getStride( equelle::Dimension::y );
    BOOST_REQUIRE_EQUAL( stride_y, dim_x + 2*ghostWidth );
}

/**
 * Test that cellAt gives the correct data.
 */
BOOST_AUTO_TEST_CASE( cellAtTest ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 1;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );

    // Collection of scalar with number of elements = (dim_x + 2*ghost) * (dim_y + 2*ghost)
    equelle::CartesianGrid::CartesianCollectionOfScalar u = grid.inputCellScalarWithDefault( "waveheights", 1.0 );

    BOOST_REQUIRE_EQUAL( u.size(), grid.number_of_cells_and_ghost_cells );

    for( int j = -ghostWidth; j < dim_y+ghostWidth; ++j ) {
        for( int i = -ghostWidth; i < dim_x+ghostWidth; ++i ) {
            //Outside domain
            if (i < 0 || j < 0) {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 0.0 );
            }
            //Outside domain
            else if (i >= dim_x || j >= dim_y) {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 0.0 );
            }
            //Inside domain
            else {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 1.0 );
            }
        }
    }
}

/**
 * Test that faceAt gives the correct data
 */
BOOST_AUTO_TEST_CASE( faceAtTest ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 2;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );
    equelle::CartesianGrid::CartesianCollectionOfScalar flux = grid.inputFaceScalarWithDefault( "flux", 0.5 );

    typedef equelle::CartesianGrid::Face Face;

    //check that a face can be reached from its two adjacent cells
    for ( int j = -ghostWidth; j < dim_y+ghostWidth; ++j ) {
        for( int i = -ghostWidth; i < dim_x+ghostWidth; ++i ) {
            BOOST_CHECK_EQUAL( &grid.faceAt( i, j, Face::negX, flux ),
                               &grid.faceAt( i-1, j, Face::posX, flux ) );

            BOOST_CHECK_EQUAL( &grid.faceAt( i, j, Face::negY, flux ),
                               &grid.faceAt( i, j-1, Face::posY, flux ) );
        }
    }

    // Check that we are zero on the ghost faces.
    for( int j = -ghostWidth; j < dim_y+ghostWidth; ++j ) {
        for( int i = -ghostWidth; i < dim_x+ghostWidth; ++i ) {
            //Outside domain
            if (i < 0 || i > dim_x) {
                BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negX, flux ), 0.0 );
            }
            else {
                //Outside domain
                if (j <0 || j >= dim_y) {
                    BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negX, flux ), 0.0 );
                }
                //Inside domain
                else {
                    BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negX, flux ), 0.5 );
                }
            }

            //Outside domain
            if (j < 0 || j > dim_y) {
                BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negY, flux ), 0.0 );
            }
            else {
                //Outside domain
                if (i <0 || i >= dim_x) {
                    BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negY, flux ), 0.0 );
                }
                //Inside domain
                else {
                    BOOST_CHECK_EQUAL( grid.faceAt( i, j, Face::negY, flux ), 0.5 );
                }
            }
        }
    }

    //Sum over all faces for a cell
    for( int j = 0; j < dim_y; ++j ) {
        for( int i = 0; i < dim_x; ++i ) {
            double sum = 0.0f;
            sum += grid.faceAt( i, j, equelle::CartesianGrid::Face::negX, flux );
            sum += grid.faceAt( i, j, equelle::CartesianGrid::Face::posX, flux );
            sum += grid.faceAt( i, j, equelle::CartesianGrid::Face::negY, flux );
            sum += grid.faceAt( i, j, equelle::CartesianGrid::Face::posY, flux );
            BOOST_CHECK_EQUAL( sum, 2.0 );
        }
    }
}

/**
 * Test that we can solve the heat equation
 */
BOOST_AUTO_TEST_CASE( heatEquation ) {
    int dim_x = 3;
    int dim_y = 5;
    int ghostWidth = 2;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );

    BOOST_CHECK_EQUAL( grid.cartdims[0], dim_x );
    BOOST_CHECK_EQUAL( grid.cartdims[1], dim_y );
    BOOST_CHECK_EQUAL( grid.dimensions, 2 );

    BOOST_CHECK_EQUAL( grid.number_of_cells, dim_x*dim_y );

    BOOST_CHECK_EQUAL( grid.number_of_cells_and_ghost_cells, (dim_x+2*ghostWidth)*(dim_y+2*ghostWidth) );

    // Collection of scalar with number of elements = (dim_x + 2*ghost) * (dim_y + 2*ghost)
    equelle::CartesianGrid::CartesianCollectionOfScalar u = grid.inputCellScalarWithDefault( "u", 1.0 );


    BOOST_REQUIRE_EQUAL( u.size(), grid.number_of_cells_and_ghost_cells );

    for( int j = -ghostWidth; j < dim_y+ghostWidth; ++j ) {
        for( int i = -ghostWidth; i < dim_x+ghostWidth; ++i ) {
            //Outside domain
            if (i < 0 || j < 0) {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 0.0 );
            }
            //Outside domain
            else if (i >= dim_x || j >= dim_y) {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 0.0 );
            }
            //Inside domain
            else {
                BOOST_CHECK_EQUAL( grid.cellAt( i, j, u ), 1.0 );
            }
        }
    }

    int stride_x = grid.getStride( equelle::Dimension::x );
    BOOST_REQUIRE_EQUAL( stride_x, 1 );

    int stride_y = grid.getStride( equelle::Dimension::y );
    BOOST_REQUIRE_EQUAL( stride_y, dim_x + 2*ghostWidth );

    const double k = 1.0; //Material specific heat diffusion constant
    const double dx = 1.0;//5.0 / static_cast<double>(dim_x);
    const double dy = 1.0;//5.0 / static_cast<double>(dim_y);
    const double dt = 0.1;

    const float a = k * dt / (dx*dy);

    double t_end = 10.0;
    double t = 0.0;

    equelle::CartesianGrid::CartesianCollectionOfScalar u0 = u;
    while (t < t_end) {
        for( int j = 0; j < dim_y; ++j ) {
            for( int i = 0; i < dim_x; ++i ) {
                grid.cellAt( i, j, u ) = grid.cellAt( i+0, j+0, u0 ) +
                                         a * 1.0/8.0 * ( grid.cellAt( i+0, j-1, u0 ) +
                                                         grid.cellAt( i+0, j+1, u0 ) +
                                                         grid.cellAt( i-1, j+0, u0 ) +
                                                         grid.cellAt( i+1, j+0, u0 ) -
                                                     4.0*grid.cellAt( i+0, j+0, u0 ) );
            }
        }
        t = t + dt;

        std::stringstream filename;
        std::showpoint(filename);
        filename << "waveheights_" << boost::format("%011.5f") % t << ".csv";
        //filename << "waveheights_" << std::setw(10) << std::setfill('0') << t << ".csv";
        std::ofstream f(filename.str());
        grid.dumpGrid( u, f );

        u0 = u;
    }
}
