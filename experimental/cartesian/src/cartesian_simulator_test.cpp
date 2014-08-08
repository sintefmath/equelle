#define BOOST_TEST_NO_MAIN

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
 * Test that we can solve the heat equation
 */
BOOST_AUTO_TEST_CASE( heatEquation ) {
	using namespace equelle;

	Opm::parameter::ParameterGroup param;

	param.insertParameter("nx", "3");
	param.insertParameter("ny", "5");
	param.insertParameter("ghost_width", "1");

    std::vector<double> init_timesteps = {{0.5, 0.5, 0.5, 0.5}};
    injectMockData( param, "timesteps", init_timesteps.begin(), init_timesteps.end() );

    std::vector<double> init_u_initial =
    {{
    		1, 1, 1,
    		1, 2, 1,
    		1, 2, 1,
    		1, 2, 1,
    		1, 1, 1,
    }};
    injectMockData( param, "u_initial", init_u_initial.begin(), init_u_initial.end() );

	equelle::CartesianEquelleRuntime er_cart(param);
    equelle::EquelleRuntimeCPU er(param);

    const Scalar k = er.inputScalarWithDefault("k", double(1));
    const Scalar dx = er.inputScalarWithDefault("dx", double(1));
    const Scalar dy = er.inputScalarWithDefault("dy", double(1));
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    const StencilCollOfScalar u_initial = er_cart.inputStencilCollectionOfScalar("u_initial", er.allCells());
    StencilCollOfScalar u0;
    u0 = u_initial;
    StencilCollOfScalar u;
    u = u_initial;
    // Note: const StencilI i = er_cart.stencilI();
    // Note: const StencilJ j = er_cart.stencilJ();
    Scalar t;
    t = double(0);
    for (const Scalar& dt : timesteps) {
        const Scalar a = (k * (dt / (dx * dy)));
        { //Start of stencil-lambda
            auto cell_stencil = [&]( int i, int j ) {
                u.grid.cellAt(u, i, j) =
                    (u0.grid.cellAt(u0, i, j) - ((a * (double(1) / double(8))) * (((((double(4) * u0.grid.cellAt(u0, i, j)) - u0.grid.cellAt(u0, (i + double(1)), j)) - u0.grid.cellAt(u0, (i - double(1)), j)) - u0.grid.cellAt(u0, i, (j + double(1)))) - u0.grid.cellAt(u0, i, (j - double(1))))));
            };
            u.grid.allCells().execute( cell_stencil );
        } // End of stencil-lambda
        t = (t + dt);
        er.output("t", t);
        er_cart.output("u", u);
        u0 = u;
    }
}

#if 0

/**
 * Test that we can solve the heat equation
 */
BOOST_AUTO_TEST_CASE( heatEquation2ndOrder ) {
    int dim_x = 30;
    int dim_y = 50;
    int ghostWidth = 2;

    typedef equelle::CartesianGrid::Face Face;

    equelle::CartesianGrid grid( std::make_tuple( dim_x, dim_y),  ghostWidth );
    equelle::CartesianGrid::CartesianCollectionOfScalar u = grid.inputCellScalarWithDefault( "u", 1.0 );
    equelle::CartesianGrid::CartesianCollectionOfScalar u_faces = grid.inputFaceScalarWithDefault( "u_faces", 0.0 );

    // const double k = 1.0; //Material specific heat diffusion constant
    // const double dx = 1.0;//5.0 / static_cast<double>(dim_x);
    // const double dy = 1.0;//5.0 / static_cast<double>(dim_y);
    const double dt = 1.0;
    // const float a = k * dt / (dx*dy);

    double t_end = 100.0;
    double t = 0.0;

    equelle::CartesianGrid::CellRange allCells = grid.allCells();
    equelle::CartesianGrid::FaceRange allXFaces = grid.allXFaces();
    equelle::CartesianGrid::FaceRange allYFaces = grid.allYFaces();

    equelle::CartesianGrid::CartesianCollectionOfScalar u0 = u;

    //Our stencil for cells
    auto cell_stencil = [&] (int i, int j) {
        grid.cellAt( i, j, u ) = //grid.cellAt( i, j, u0 ) +l
                     1.0/4.0 * ( grid.faceAt(i, j, Face::negX, u_faces) +
                                 grid.faceAt(i, j, Face::posX, u_faces) +
                                 grid.faceAt(i, j, Face::negY, u_faces) +
                                 grid.faceAt(i, j, Face::posY, u_faces) );
    };

    auto x_face_stencil = [&] (int i, int j) {
        grid.faceAt( i, j, Face::negX, u_faces ) =
                    1.0 / 6.0 * ( grid.cellAt( i-2, j, u0 ) +
                            2.0 * grid.cellAt( i-1, j, u0 )  +
                            2.0 * grid.cellAt( i, j, u0 )  +
                                  grid.cellAt( i+1, j, u0 ) );
    };

    auto y_face_stencil = [&] (int i, int j) {
        grid.faceAt( i, j, Face::negY, u_faces ) =
                1.0 / 6.0 * ( grid.cellAt( i, j-2, u0 ) +
                        2.0 * grid.cellAt( i, j-1, u0 )  +
                        2.0 * grid.cellAt( i, j, u0 )  +
                              grid.cellAt( i, j+1, u0 ) );
    };

    while (t < t_end) {
        allXFaces.execute(x_face_stencil);
        allYFaces.execute(y_face_stencil);
        allCells.execute(cell_stencil);

        t = t + dt;
        u0 = u;

        std::stringstream filename;
        std::showpoint(filename);
        filename << "waveheights_" << boost::format("%011.5f") % t << ".csv";

        std::ofstream file(filename.str());
        grid.dumpGridCells( u, file );
    }
}

#endif
