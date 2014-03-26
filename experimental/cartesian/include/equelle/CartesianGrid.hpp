#pragma once

#include <array>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <map>

#include <opm/core/utility/parameters/ParameterGroup.hpp>

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
     * @brief CartesianGrid
     * @param param Is a parameter object where the following keys are used for grid initialization.
     *              - grid_dim Dimension of grid. (default 2)
     *              - nx Number of interior cells in x-direction. (default 3)
     *              - ny Number of interior cells in y-direction. (default 2)
     *              - ghost_width width of ghost boundary. (default 1)
     *              In addition how to read initial and boundary conditions can be specified.
     */
    CartesianGrid( const Opm::parameter::ParameterGroup& param );

    /**
     * @brief CartesianGrid constructor for 2D-grids.
     * @param dims number of cells in x and y dimension.
     * @param ghostWidth width of ghost boundary. Assumed to be uniform in both directions.
     */
    explicit CartesianGrid(std::tuple<int, int> dims, int ghostWidth );

    ~CartesianGrid();

    std::array<int, 2> cartdims{{-1,-1}}; //! Number of interior cells in each dimension.
    typedef std::array<int, 2> strideArray;
    strideArray cellStrides;

    std::map<Dimension, strideArray> faceStrides;
    std::map<Dimension, int>         number_of_faces_with_ghost_cells;

    int dimensions;            //! Number of spatial dimensions.
    int number_of_cells;       //! Number of interior cells in the grid.
    int ghost_width; //! Number of ghost cells. Assumed to be the same for all directions and on every side of the domain.
    int number_of_cells_and_ghost_cells;
    //int number_of_faces_and_ghost_faces;

    typedef std::vector<double> CartesianCollectionOfScalar;


    CartesianCollectionOfScalar inputCellCollectionOfScalar( std::string name );
    CartesianCollectionOfScalar inputFaceCollectionOfScalar( std::string name );


    CartesianCollectionOfScalar inputCellScalarWithDefault( std::string name, double d );
    CartesianCollectionOfScalar inputFaceScalarWithDefault( std::string name, double d );

    int getStride( Dimension );

    double& cellAt( int i, int j, CartesianCollectionOfScalar& coll );
    double& faceAt( int i, int j, Face, CartesianCollectionOfScalar& coll );

    /**
     * @brief dumpGrid a grid to a stream or file.
     * @param grid
     * @param stream
     */
    void dumpGrid( const CartesianCollectionOfScalar& grid, std::ostream& stream );

private:
    const Opm::parameter::ParameterGroup param_;
    void init2D( std::tuple<int, int> dims, int ghostWidth );
};

} // namespace equelle
