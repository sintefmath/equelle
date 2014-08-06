#pragma once

#include <array>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <map>
#include <functional>

#include "equelle/equelleTypes.hpp"

#include <opm/core/utility/parameters/ParameterGroup.hpp>

namespace equelle {

enum Dimension {
    x = 0,
    y = 1,
    z = 2
};

class CartesianCollOfCell;

class CartesianEquelleRuntime {
public:
    /**
     * @brief CartesianGrid constructor for a parameter object.
     * @param param Is a parameter object where the following keys are used for grid initialization.
     *              - grid_dim Dimension of grid. (default 2)
     *              - nx Number of interior cells in x-direction. (default 3)
     *              - ny Number of interior cells in y-direction. (default 2)
     *              - ghost_width width of ghost boundary. (default 1)
     *              In addition how to read initial and boundary conditions can be specified.
     */
	CartesianEquelleRuntime( const Opm::parameter::ParameterGroup& param );

	CartesianCollOfCell inputCellCollectionOfScalar( std::string name );
	//CartesianCollOfFace inputFaceCollectionOfScalar( std::string name );

	CartesianCollOfCell inputCellScalarWithDefault( std::string name, double d );
	//CartesianCollOfFace inputFaceScalarWithDefault( std::string name, double d );
private:
    const Opm::parameter::ParameterGroup param_;
};

/**
 * @brief The CartesianGrid class models Opm::UnstructuredGrid in spirit, but is tailored for cartesian dense grids.
 */
class CartesianGrid {
public:
    class CellRange;
    class FaceRange;

    /**
     * @brief The Face enum is used to indicate which adjacent face one is referring to for a given cell id.
     *
     *        This enum is used as a parameter to faceAt, and can be interprented as a way of "half-index", thus
     *        allowing for staggered grids.
     */
    enum class Face {
        negX, posX, negY, posY, negZ, posZ
    };

    typedef std::array<int, 2> strideArray;

    CartesianGrid();
    ~CartesianGrid();

    /**
     * @brief CartesianGrid constructor for 2D-grids.
     * @param dims number of cells in x and y dimension.
     * @param ghostWidth width of ghost boundary. Assumed to be uniform in both directions.
     */
    explicit CartesianGrid(std::tuple<int, int> dims, int ghostWidth );


    std::array<int, 2> cartdims{{-1,-1}}; //!< Number of interior cells in each dimension.
    strideArray cellStrides;

    std::array<strideArray, 2>       faceStrides;
    std::array<int, 2 >              number_of_faces_with_ghost_cells;

    int dimensions;            //!< Number of spatial dimensions.
    int number_of_cells;       //!< Number of interior cells in the grid.
    int ghost_width;           //!< Width of ghost cell boundary. Assumed to be the same for all directions and on every side of the domain.
    int number_of_cells_and_ghost_cells; //!< Total number of cells and ghost cells in grid.


    /**
     * @brief cellAt Return a reference to an element of cell(i,j)
     * @param i Index i of cell
     * @param j Index j of cell
     * @param coll  A scalar collection representing face values in the grid.
     * @return The value of the collection at the given edge.
     */
    double& cellAt( int i, int j, CartesianCollOfCell& coll ) const;
    const double& cellAt( int i, int j, const CartesianCollOfCell& coll ) const ;

    /**
     * @brief faceAt Return a reference to an element of a face adjacent to cell (i,j).
     *
     * The method is intended both for reading from and writing to a variable.
     * @param i Index i of cell
     * @param j Index j of cell
     * @param face Id of which adjacent face one is referring to.
     * @param coll A scalar collection representing face values in the grid.
     * @return The value of the collection at the given face.
     * /
    double& faceAt( int i, int j, Face face, CartesianCollOfFace& coll ) const;
    const double& faceAt( int i, int j, Face face, const CartesianCollOfFace& coll ) const;
    */

    /**
     * @brief dumpGrid a grid to a stream or file.
     * @param grid
     * @param stream
     */
    void dumpGridCells( const CartesianCollOfCell& grid, std::ostream& stream );
    //void dumpGridFaces( /*const*/ CartesianCollectionOfScalar& grid, Face, std::ostream& stream );

    /**
     * Returns an object that can execute a stencil on all cells/faces within a given range
     */
    CellRange allCells();
    FaceRange allXFaces();
    FaceRange allYFaces();

private:
    void init2D( std::tuple<int, int> dims, int ghostWidth );
    int cellOrigin;
};





/**
 * Class that enables execution of a stencil on all cells within the range
 */
class CartesianGrid::CellRange {
public:
    CellRange(int i0, int i1, int j0, int j1)
        : i_begin(i0), i_end(i1), j_begin(j0), j_end(j1)
    {

    }

    void execute(std::function<void(int, int)> stencil)
    {
//#pragma omp parallel for here for parallelism
        for (int j=j_begin; j < j_end; ++j) {
            for (int i=i_begin; i < i_end; ++i) {
                stencil(i, j);
            }
        }
    }

private:
    int i_begin;
    int i_end;

    int j_begin;
    int j_end;
};

/**
 * Same as a cellrange, but it loops over all faces
 */
class CartesianGrid::FaceRange {
public:
    FaceRange(int i0, int i1, int j0, int j1)
        : i_begin(i0), i_end(i1), j_begin(j0), j_end(j1)
    {

    }

    void execute(std::function<void(int, int)> stencil)
    {
//#pragma omp parallel for here for parallelism
        for (int j=j_begin; j < j_end; ++j) {
            for (int i=i_begin; i < i_end; ++i) {
                stencil(i, j);
            }
        }
    }

private:
    int i_begin;
    int i_end;

    int j_begin;
    int j_end;
};


//FIXME Need also CartesianCollOfFace
class CartesianCollOfCell {
public:
	CartesianCollOfCell(std::tuple<int, int> dims, int ghostWidth, double default_value=0.0f)
    	: grid(dims, ghostWidth)
    {
    	data.resize(grid.number_of_cells_and_ghost_cells, 0.0f);
    	if (default_value != 0.0f) {
    		for (int j=0; j<std::get<1>(dims); ++j) {
    			double* begin = &data[(j+ghostWidth)*(std::get<0>(dims) + 2*ghostWidth) + ghostWidth];
    			double* end = begin + std::get<0>(dims);
				std::fill(begin, end, default_value);
    		}
    	}
	}

    std::vector<double> data;
    CartesianGrid grid;

private:
};


} // namespace equelle
