#pragma once

#include <array>
#include <vector>
#include <tuple>

namespace equelle {

enum class Dimension {
    x = 0,
    y = 1,
    z = 2
};



/**
 * @brief The CartesianGrid class models Opm::UnstructuredGrid in spirit, but is tailored for cartesian dense grids.
 */
class CartesianGrid {
public:
    enum class Face {
        negX, posX, negY, posY, negZ, posZ
    };

    CartesianGrid();
    /**
     * @brief CartesianGrid constructor for 2D-grids.
     * @param dims number of cells in x and y dimension.
     * @param ghostWidth width of ghost boundary. Assumed to be uniform in both directions.
     */
    explicit CartesianGrid(std::tuple<int, int> dims, int ghostWidth );

    ~CartesianGrid();

    std::array<int, 2> cartdims{{-1,-1}}; //! Number of interior cells in each dimension.
    std::array<int, 2> cellStrides;
    std::array<int, 2> faceStrides;

    int dimensions;            //! Number of spatial dimensions.
    int number_of_cells;       //! Number of interior cells in the grid.
    int ghost_width; //! Number of ghost cells. Assumed to be the same for all directions and on every side of the domain.
    int number_of_cells_and_ghost_cells;
    int number_of_faces_and_ghost_faces;

    typedef std::vector<double> CartesianCollectionOfScalar;

    CartesianCollectionOfScalar inputCellScalarWithDefault( std::string name, double d );
    CartesianCollectionOfScalar inputFaceScalarWithDefault( std::string name, double d );

    int getStride( Dimension );

    double& cellAt( int i, int j, CartesianCollectionOfScalar& coll );
    //double* faceAt( int i, int j, std::tuple<Face, Face>, CartesianCollectionOfScalar& coll );

    /**
     * @brief dumpGrid a grid to a stream or file.
     * @param grid
     * @param stream
     */
    void dumpGrid( const CartesianCollectionOfScalar& grid, std::ostream& stream );


};

} // namespace equelle
