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
    CartesianGrid();
    explicit CartesianGrid(std::tuple<int, int> dims, int numGhost );
    ~CartesianGrid();

    std::array<int, 3> cartdims{{-1,-1,-1}}; //! Number of interior cells in each dimension.
    std::array<int, 3> strides;

    int dimensions;            //! Number of spatial dimensions.
    int number_of_cells;       //! Number of interior cells in the grid.
    int ghost_width; //! Number of ghost cells. Assumed to be the same for all directions and on every side of the domain.
    int number_of_cells_and_ghost_cells;

    typedef std::vector<double> CartesianCollectionOfScalar;

    CartesianCollectionOfScalar inputCellScalarWithDefault( std::string name, double d );

    int getStride( Dimension );

    double* cellAt( int i, int j, CartesianCollectionOfScalar& coll );

    /**
     * @brief dumpGrid a grid to a stream or file.
     * @param grid
     * @param stream
     */
    void dumpGrid( const CartesianCollectionOfScalar& grid, std::ostream& stream );


};

} // namespace equelle
