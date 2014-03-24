#include <algorithm>
#include <numeric>
#include <cassert>

#include "equelle/CartesianGrid.hpp"

equelle::CartesianGrid::CartesianGrid()
{

}

equelle::CartesianGrid::CartesianGrid(std::tuple<int, int> dims, int numGhost )
{
    cartdims[0] = std::get<0>( dims );
    cartdims[1] = std::get<1>( dims );

    strides[0] = 1;
    strides[1] = 2*numGhost + cartdims[1];
    strides[2] = -1;

    this->ghost_width = numGhost;
    this->dimensions = 2;
    this->number_of_cells = cartdims[0]*cartdims[1];

    this->number_of_cells_and_ghost_cells = (cartdims[0]+2*numGhost) * (cartdims[1]+2*numGhost);
}

equelle::CartesianGrid::~CartesianGrid()
{

}

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputCellScalarWithDefault(std::string name, double d)
{
    CartesianCollectionOfScalar v( number_of_cells_and_ghost_cells, 0.0 );

    for( int j = 0; j < cartdims[1]; ++j ) {
        for( int i = 0; i < cartdims[0]; ++i ) {
            *cellAt( i, j, v ) = d;
        }
    }

    return v;
}

int equelle::CartesianGrid::getStride(equelle::Dimension)
{
    return 0;
}

double *equelle::CartesianGrid::cellAt( int i, int j, equelle::CartesianGrid::CartesianCollectionOfScalar &coll )
{
    int origin = ghost_width * strides[1] + ghost_width * strides[0];
    return &coll[ origin + j*strides[1] + i*strides[0] ];
}

