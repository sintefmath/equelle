#include <algorithm>
#include <numeric>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <iomanip>

#include "equelle/CartesianGrid.hpp"

equelle::CartesianGrid::CartesianGrid()
{

}

equelle::CartesianGrid::CartesianGrid(std::tuple<int, int> dims, int numGhost )
{
    cartdims[0] = std::get<0>( dims );
    cartdims[1] = std::get<1>( dims );

    strides[0] = 1;
    strides[1] = 2*numGhost + cartdims[0];
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

int equelle::CartesianGrid::getStride(equelle::Dimension dim)
{
    switch (dim) {
    case equelle::Dimension::x:
        return strides[0];
        break;
    case equelle::Dimension::y:
        return strides[1];
        break;
    case equelle::Dimension::z:
        return strides[2];
        break;
    default:
        throw std::runtime_error( "Trying to get stride for nonexisten dimension" );
    }
}

double *equelle::CartesianGrid::cellAt( int i, int j, equelle::CartesianGrid::CartesianCollectionOfScalar &coll )
{
    int origin = ghost_width * strides[1] + ghost_width * strides[0];

    int index = origin + j*strides[1] + i*strides[0];
    return &coll[ index ];
}

void equelle::CartesianGrid::dumpGrid(const equelle::CartesianGrid::CartesianCollectionOfScalar &grid, std::ostream &stream)
{
    for( int j = 0; j < cartdims[1] + 2*ghost_width; ++j ) {
        std::copy_n( grid.begin() + j*strides[1], cartdims[0] + 2*ghost_width, std::ostream_iterator<double>( stream, "," ) );
        stream << std::endl;
    }
}

