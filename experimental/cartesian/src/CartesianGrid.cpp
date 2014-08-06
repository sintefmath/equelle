#include <algorithm>
#include <numeric>
#include <cassert>
#include <iterator>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <iostream>

#include <opm/autodiff/AutoDiffHelpers.hpp>


#include "equelle/CartesianGrid.hpp"
#include "equelle/equelleTypes.hpp"

equelle::CartesianGrid::CartesianGrid()
{

}

equelle::CartesianEquelleRuntime::CartesianEquelleRuntime(const Opm::parameter::ParameterGroup &param)
    : param_( param )
{
    int grid_dim = param.getDefault( "grid_dim", 2 );
    if ( grid_dim != 2 ) {
        throw std::runtime_error( "Only 2D-cartesian grids are supported, so far..." );
    }
}

equelle::CartesianCollOfCell equelle::CartesianEquelleRuntime::inputCellCollectionOfScalar(std::string name)
{
    std::tuple<int, int> dims;
    param_.get( "nx", std::get<0>(dims) );
    param_.get( "ny", std::get<1>(dims) );
    int ghostWidth = param_.getDefault( "ghost_width", 1 );

    CartesianCollOfCell v(dims, ghostWidth);

    const bool from_file = param_.getDefault(name + "_from_file", false);
    if ( from_file ) {
        const String filename = param_.get<String>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            OPM_THROW(std::runtime_error, "Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;

        for( int j = 0; j < std::get<1>(dims); ++j ) {
            for( int i = 0; i < std::get<0>(dims); ++i ) {
                if ( beg == end ) {
                    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
                }
                v.grid.cellAt( i, j, v ) = *beg;
                beg++;
            }
        }
        return v;
    } else { // Constant value
        const double value = param_.get<double>( name );
        return inputCellScalarWithDefault( name, value );
    }
}


/*
equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputFaceCollectionOfScalar(std::string name)
{
    CartesianCollectionOfScalar v;

    const bool from_file = param_.getDefault(name + "_from_file", false);
    if ( from_file ) {
        const String filename = param_.get<String>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            OPM_THROW(std::runtime_error, "Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;

        int num = number_of_faces_with_ghost_cells[Dimension::x] * (cartdims[1]+2*ghost_width) +
                number_of_faces_with_ghost_cells[Dimension::y] * (cartdims[0]+2*ghost_width);

        v.resize( num, 0.0 );

        // X-faces
        for( int j = 0; j < cartdims[1]; ++j ) {
            for( int i = 0; i <= cartdims[0]; ++i ) {
                if ( beg == end ) {
                    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
                }
                faceAt( i, j, Face::negX, v ) = *beg;
                beg++;
            }
        }

        // Y-faces        
        // NB. Here we have switch the order we traverse the dimensions, in order to allow for
        // the natural indexing of storing y-data in input files.
        for( int i = 0; i < cartdims[0]; ++i ) {
            for( int j = 0; j <= cartdims[1]; ++j ) {
                if ( beg == end ) {
                    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
                }
                faceAt( i, j, Face::negY, v ) = *beg;
                beg++;
            }
        }
    } else { // Constant value
        const double value = param_.get<double>( name );
        v = inputFaceScalarWithDefault( name, value );
    }

    return v;
}
*/

equelle::CartesianCollOfCell equelle::CartesianEquelleRuntime::inputCellScalarWithDefault(std::string /*name*/, double d)
{    
    std::tuple<int, int> dims;
    param_.get( "nx", std::get<0>(dims) );
    param_.get( "ny", std::get<1>(dims) );
    int ghostWidth = param_.getDefault( "ghost_width", 1 );

    return CartesianCollOfCell(dims, ghostWidth, d);
}

/*
equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputFaceScalarWithDefault(std::string , double d )
{
    int num = number_of_faces_with_ghost_cells[Dimension::x] * (cartdims[1]+2*ghost_width) +
            number_of_faces_with_ghost_cells[Dimension::y] * (cartdims[0]+2*ghost_width);

    CartesianCollectionOfScalar v( num, 0.0 );

    for( int j = 0; j < cartdims[1]; ++j ) {
        for( int i = 0; i <= cartdims[0]; ++i ) {
            faceAt( i, j, Face::negX, v ) = d;
        }
    }

    for( int j = 0; j <= cartdims[1]; ++j ) {
        for( int i = 0; i < cartdims[0]; ++i ) {
            faceAt( i, j, Face::negY, v ) = d;
        }
    }

    return v;
}
*/
















void equelle::CartesianGrid::init2D( std::tuple<int, int> dims, int ghostWidth )
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
    this->cellOrigin = ghost_width * cellStrides[1] + ghost_width * cellStrides[0];

    number_of_faces_with_ghost_cells[Dimension::x] = (cartdims[0]+2*ghostWidth+1);
    number_of_faces_with_ghost_cells[Dimension::y] = (cartdims[1]+2*ghostWidth+1);
}

equelle::CartesianGrid::CartesianGrid( std::tuple<int, int> dims, int ghostWidth )
{
    init2D( dims, ghostWidth );
}

equelle::CartesianGrid::~CartesianGrid()
{

}




double& equelle::CartesianGrid::cellAt( const int i, const int j, equelle::CartesianCollOfCell &coll ) const
{
    const int index = cellOrigin + j*cellStrides[1] + i*cellStrides[0];
    return coll.data[ index ];
}

const double &equelle::CartesianGrid::cellAt( const int i, const int j, const equelle::CartesianCollOfCell &coll) const
{
    const int index = cellOrigin + j*cellStrides[1] + i*cellStrides[0];
    return coll.data[ index ];
}

/*
double &equelle::CartesianGrid::faceAt(int i, int j, const equelle::CartesianGrid::Face face, equelle::CartesianGrid::CartesianCollectionOfScalar &coll) const
{
    i += ghost_width;
    j += ghost_width;

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
        break;
    default:
        throw std::runtime_error("Only 2D-cartesian grids are supported, so far...");
        break;
    }

    return coll[offset + j*strides[1] + i*strides[0] ];
}

const double &equelle::CartesianGrid::faceAt(int i, int j, equelle::CartesianGrid::Face face, const equelle::CartesianGrid::CartesianCollectionOfScalar &coll) const
{
    i += ghost_width;
    j += ghost_width;

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
        break;
    default:
        throw std::runtime_error("Only 2D-cartesian grids are supported, so far...");
        break;
    }

    return coll[offset + j*strides[1] + i*strides[0] ];
}
*/

void equelle::CartesianGrid::dumpGridCells(const equelle::CartesianCollOfCell &cells, std::ostream &stream)
{
    int num_columns = cartdims[0] + 2*ghost_width;
    for( int j = 0; j < cartdims[1] + 2*ghost_width; ++j ) {
        int row_offset  = j*cellStrides[1];
        std::copy_n( cells.data.begin() + row_offset, num_columns - 1, std::ostream_iterator<double>( stream, "," ) );
        stream << cells.data[row_offset + num_columns-1];
        stream << std::endl;
    }
}


/*
void equelle::CartesianGrid::dumpGridFaces( equelle::CartesianGrid::CartesianCollectionOfScalar &faces, Face input_face, std::ostream &stream)
{
    Face face = input_face;
    int face_offset_x = 0;
    int face_offset_y = 0;

    switch(face) {
    case Face::posX:
    case Face::negX:
        face = Face::negX;
        face_offset_x = 1;
        break;
    case Face::posY:
    case Face::negY:
        face = Face::negY;
        face_offset_y = 1;
        break;
    default:
        throw std::runtime_error("Unknown face position");
    }

    int start_x = -ghost_width;
    int end_x = cartdims[0] + ghost_width + face_offset_x;

    int start_y = -ghost_width;
    int end_y = cartdims[1] + ghost_width + face_offset_y;


    for( int j = start_y; j < end_y; ++j ) {
        stream << faceAt( start_x, j, face, faces);
        for( int i = start_x+1; i < end_x; ++i ) {
            stream << "," << faceAt( i, j, face, faces);
        }
        stream << std::endl;
    }
}
*/

equelle::CartesianGrid::CellRange equelle::CartesianGrid::allCells() {
    return CellRange(0, cartdims[0], 0, cartdims[1]);
}

equelle::CartesianGrid::FaceRange equelle::CartesianGrid::allXFaces() {
    return FaceRange(0, cartdims[0]+1, 0, cartdims[1]);
}

equelle::CartesianGrid::FaceRange equelle::CartesianGrid::allYFaces() {
    return FaceRange(0, cartdims[0], 0, cartdims[1]+1);
}

