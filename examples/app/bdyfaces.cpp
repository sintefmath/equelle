/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include <opm/grid/GridManager.hpp>
#include <opm/grid/UnstructuredGrid.h>
#include <iostream>

int main(int /*argc*/, char** argv)
{
    Opm::GridManager gm(argv[1]);
    const UnstructuredGrid& g = *(gm.c_grid());
    enum { Outside = -1 };
    double minx = 1e100;
    double maxx = -1e100;
    for (int f = 0; f < g.number_of_faces; ++f) {
        const double fcx = g.face_centroids[3*f];
        minx = std::min(minx, fcx);
        maxx = std::max(maxx, fcx);
    }
    const double xlim = minx + 0.98 * (maxx - minx);
    for (int f = 0; f < g.number_of_faces; ++f) {
        if (g.face_cells[2*f] == Outside || g.face_cells[2*f + 1] == Outside) {
            // Boundary face
            if (g.face_centroids[3*f] > xlim) {
                std::cout << f << '\n';
            }
        }
    }
}
