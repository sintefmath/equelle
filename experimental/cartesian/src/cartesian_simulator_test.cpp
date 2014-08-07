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





/**
 * Test that we can solve the heat equation
 */
BOOST_AUTO_TEST_CASE( heatEquation ) {
	Opm::parameter::ParameterGroup param;

	param.insertParameter("nx", "30");
	param.insertParameter("ny", "50");
	param.insertParameter("ghost_width", "1");

	equelle::CartesianEquelleRuntime er_cart(param);

    equelle::CartesianCollOfCell u = er_cart.inputCellScalarWithDefault( "u", 1.0 );

    const double k = 1.0; //Material specific heat diffusion constant
    const double dx = 1.0;//5.0 / static_cast<double>(dim_x);
    const double dy = 1.0;//5.0 / static_cast<double>(dim_y);
    const double dt = 1.0;

    const float a = k * dt / (dx*dy);

    double t_end = 100.0;
    double t = 0.0;

    equelle::CartesianGrid::CellRange allCells = u.grid.allCells();

    equelle::CartesianCollOfCell u0 = u;

    while (t < t_end) {
        //Our stencil for cells
        auto cell_stencil = [&] (int i, int j) {
            u.grid.cellAt( i, j, u ) = u0.grid.cellAt( i, j, u0 ) +
                      a* 1.0/8.0 * ( u0.grid.cellAt(i+1, j, u0) +
                    		         u0.grid.cellAt(i-1, j, u0) +
                    		         u0.grid.cellAt(i, j+1, u0) +
                    		         u0.grid.cellAt(i, j-1, u0) -
                                     4*u0.grid.cellAt(i, j, u0) );
        };
        allCells.execute(cell_stencil);

        t = t + dt;
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
