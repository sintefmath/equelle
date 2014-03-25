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

equelle::CartesianGrid::CartesianGrid(std::tuple<int, int> dims, int ghostWidth )
{
    cartdims[0] = std::get<0>( dims );
    cartdims[1] = std::get<1>( dims );

    cellStrides[0] = 1;
    cellStrides[1] = 2*ghostWidth + cartdims[0];

    faceStrides[Dimension::x] = {{1, cellStrides[1] + 1}};
    faceStrides[Dimension::y] = {{1, cellStrides[1]}};


    this->ghost_width = ghostWidth;
    this->dimensions = 2;
    this->number_of_cells = cartdims[0]*cartdims[1];

    this->number_of_cells_and_ghost_cells = (cartdims[0]+2*ghostWidth) * (cartdims[1]+2*ghostWidth);

    number_of_faces_with_ghost_cells[Dimension::x] = (cartdims[0]+2*ghostWidth+1);
    number_of_faces_with_ghost_cells[Dimension::y] = (cartdims[1]+2*ghostWidth+1);
}

equelle::CartesianGrid::~CartesianGrid()
{

}

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputCellScalarWithDefault(std::string name, double d)
{
    CartesianCollectionOfScalar v( number_of_cells_and_ghost_cells, 0.0 );

    for( int j = 0; j < cartdims[1]; ++j ) {
        for( int i = 0; i < cartdims[0]; ++i ) {
            cellAt( i, j, v ) = d;
        }
    }

    return v;
}

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputFaceScalarWithDefault(std::string name, double d)
{
    int num = number_of_faces_with_ghost_cells[Dimension::x] * cartdims[1] +
              number_of_faces_with_ghost_cells[Dimension::y] * cartdims[0];

    CartesianCollectionOfScalar v( num, 0.0 );

    return v;
}

int equelle::CartesianGrid::getStride(equelle::Dimension dim)
{
    switch (dim) {
    case equelle::Dimension::x:
        return cellStrides[0];
        break;
    case equelle::Dimension::y:
        return cellStrides[1];
        break;
    case equelle::Dimension::z:
        return cellStrides[2];
        break;
    default:
        throw std::runtime_error( "Trying to get stride for nonexistent dimension" );
    }
}

double& equelle::CartesianGrid::cellAt( int i, int j, equelle::CartesianGrid::CartesianCollectionOfScalar &coll )
{
    int origin = ghost_width * cellStrides[1] + ghost_width * cellStrides[0];

    int index = origin + j*cellStrides[1] + i*cellStrides[0];
    return coll[ index ];
}

double &equelle::CartesianGrid::faceAt(int i, int j, equelle::CartesianGrid::Face face, equelle::CartesianGrid::CartesianCollectionOfScalar &coll)
{
    // Update index to the correct cell.
    switch (face) {
    case Face::posX:
        ++i;
        break;
    case Face::posY:
        ++j;
        break;
    default:
        // Intentional, the other faces does not need index manipulation.
        break;
    }

    int offset = 0;
    strideArray strides;

    switch (face) {
    case Face::posX:
    case Face::negX:
        offset = 0;
        strides = faceStrides[Dimension::x];
        break;
    case Face::posY:
    case Face::negY:
        offset = number_of_faces_with_ghost_cells[Dimension::x] * ( cartdims[1] + 2*ghost_width );
        strides = faceStrides[Dimension::y];
    default:
        // Intentional, the other
        break;
    }

    return coll[offset + j*strides[1] + i*strides[0] ];
}

void equelle::CartesianGrid::dumpGrid(const equelle::CartesianGrid::CartesianCollectionOfScalar &grid, std::ostream &stream)
{
    for( int j = 0; j < cartdims[1] + 2*ghost_width; ++j ) {
        std::copy_n( grid.begin() + j*cellStrides[1], cartdims[0] + 2*ghost_width, std::ostream_iterator<double>( stream, "," ) );
        stream << std::endl;
    }
}
