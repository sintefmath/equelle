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


BOOST_AUTO_TEST_CASE( cartesianRuntimeTest ) {
	Opm::parameter::ParameterGroup param;

	param.insertParameter("nx", "30");
	param.insertParameter("ny", "50");
	param.insertParameter("ghost_width", "1");

	equelle::CartesianEquelleRuntime er_cart(param);
	//What to test her actually?
}

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

    BOOST_REQUIRE_EQUAL( grid.cellStrides[0], 1 );
    BOOST_REQUIRE_EQUAL( grid.cellStrides[1], dim_x + 2*ghostWidth );
}


BOOST_AUTO_TEST_CASE( cartesianCollOfScalarTest ) {
	Opm::parameter::ParameterGroup param;

	param.insertParameter("nx", "30");
	param.insertParameter("ny", "50");
	param.insertParameter("ghost_width", "1");

	equelle::CartesianEquelleRuntime er_cart(param);

    equelle::StencilCollOfScalar u = er_cart.inputCellScalarWithDefault( "u", 1.0 );

    BOOST_REQUIRE_EQUAL( u.data.size(), u.grid.number_of_cells_and_ghost_cells );
    BOOST_REQUIRE_EQUAL( std::get<0>(u.grid.cartdims), 30 );
    BOOST_REQUIRE_EQUAL( std::get<1>(u.grid.cartdims), 50 );
    BOOST_REQUIRE_EQUAL( u.grid.ghost_width, 1 );
}

BOOST_AUTO_TEST_CASE( cellAtTest ) {
	Opm::parameter::ParameterGroup param;

	param.insertParameter("nx", "3");
	param.insertParameter("ny", "5");
	param.insertParameter("ghost_width", "1");

	const int nx = 3;
	const int ny = 5;
	const int ghost_width = 1;

	equelle::CartesianEquelleRuntime er_cart(param);

    // Collection of scalar with number of elements = (dim_x + 2*ghost) * (dim_y + 2*ghost)
    equelle::StencilCollOfScalar u = er_cart.inputCellScalarWithDefault( "waveheights", 1.0 );

    BOOST_REQUIRE_EQUAL( u.data.size(), u.grid.number_of_cells_and_ghost_cells );

    for( int j = -ghost_width; j < ny+ghost_width; ++j ) {
        for( int i = -ghost_width; i < nx+ghost_width; ++i ) {
            //Outside domain
            if (i < 0 || j < 0) {
                BOOST_CHECK_EQUAL( u.grid.cellAt( u, i, j ), 0.0 );
            }
            //Outside domain
            else if (i >= nx || j >= ny) {
                BOOST_CHECK_EQUAL( u.grid.cellAt( u, i, j ), 0.0 );
            }
            //Inside domain
            else {
                BOOST_CHECK_EQUAL( u.grid.cellAt( u, i, j ), 1.0 );
            }
        }
    }
}

#if 0
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


BOOST_AUTO_TEST_CASE( disallow3DGrids ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    // Test that we do not allow for constructions of other than 2D-grids.
    param.insertParameter( "grid_dim", "3" );
    BOOST_CHECK_THROW( equelle::CartesianGrid grid( param ), std::runtime_error  );
}

BOOST_AUTO_TEST_CASE( ctorFromParamterObject ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "grid_dim", "2");
    param.insertParameter( "nx", "10" );
    param.insertParameter( "ny", "12" );
    param.insertParameter( "ghost_width", "2" );

    equelle::CartesianGrid grid(param);
    BOOST_CHECK_EQUAL( grid.ghost_width, 2 );
    BOOST_CHECK_EQUAL( grid.dimensions, 2 );
    BOOST_CHECK_EQUAL( grid.cartdims[0], 10 );
    BOOST_CHECK_EQUAL( grid.cartdims[1], 12 );
}

BOOST_AUTO_TEST_CASE( cellDataFromFile ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "nx", "2" );
    param.insertParameter( "ny", "2" );


    std::vector<double> defaults = {{1,2,3,4}};
    injectMockData( param, "waveheights", defaults.begin(), defaults.end() );

    equelle::CartesianGrid grid(param);
    auto u = grid.inputCellCollectionOfScalar( "waveheights" );
    BOOST_CHECK_EQUAL( grid.cellAt( 0, 0, u ), 1 );
    BOOST_CHECK_EQUAL( grid.cellAt( 1, 0, u ), 2 );
    BOOST_CHECK_EQUAL( grid.cellAt( 0, 1, u ), 3 );
    BOOST_CHECK_EQUAL( grid.cellAt( 1, 1, u ), 4 );
}

BOOST_AUTO_TEST_CASE( constantCellData ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "nx", "2" );
    param.insertParameter( "ny", "2" );
    param.insertParameter( "waveheights", "42" );

    equelle::CartesianGrid grid(param);

    auto u = grid.inputCellCollectionOfScalar( "waveheights" );
    BOOST_CHECK_EQUAL( grid.cellAt( 0, 0, u ), 42 );
    BOOST_CHECK_EQUAL( grid.cellAt( 1, 0, u ), 42 );
    BOOST_CHECK_EQUAL( grid.cellAt( 0, 1, u ), 42 );
    BOOST_CHECK_EQUAL( grid.cellAt( 1, 1, u ), 42 );
}

BOOST_AUTO_TEST_CASE( constantFaceData ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "nx", "2" );
    param.insertParameter( "ny", "2" );
    param.insertParameter( "flux", "-1.5" );

    equelle::CartesianGrid grid(param);

    auto f = grid.inputFaceCollectionOfScalar( "flux" );

    BOOST_CHECK_EQUAL( grid.faceAt( 0, 0, equelle::CartesianGrid::Face::negX, f ), -1.5 );
    BOOST_CHECK_EQUAL( grid.faceAt( 1, 0, equelle::CartesianGrid::Face::negX, f ), -1.5 );
    BOOST_CHECK_EQUAL( grid.faceAt( 2, 0, equelle::CartesianGrid::Face::negX, f ), -1.5 );

    // Check that ghost face is not set.
    BOOST_CHECK_EQUAL( grid.faceAt( 2, 0, equelle::CartesianGrid::Face::posX, f ), 0.0 );
}

BOOST_AUTO_TEST_CASE( faceDataFromFile ) {
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "nx", "2" );
    param.insertParameter( "ny", "2" );

    std::vector<double> defaults = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };

    injectMockData( param, "flux", defaults.begin(), defaults.end() );

    equelle::CartesianGrid grid(param);
    auto u = grid.inputFaceCollectionOfScalar( "flux" );
    BOOST_CHECK_EQUAL( grid.faceAt( 0, 0, equelle::CartesianGrid::Face::negX, u ), 1 );
    BOOST_CHECK_EQUAL( grid.faceAt( 1, 0, equelle::CartesianGrid::Face::posX, u ), 3 );

    BOOST_CHECK_EQUAL( grid.faceAt( 1, 1, equelle::CartesianGrid::Face::negY, u ), 11 );
    BOOST_CHECK_EQUAL( grid.faceAt( 1, 1, equelle::CartesianGrid::Face::posY, u ), 12 );


}
#endif
