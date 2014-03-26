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

equelle::CartesianGrid::CartesianGrid(const Opm::parameter::ParameterGroup &param)
    : param_( param )
{
    int grid_dim = param.getDefault( "grid_dim", 2 );
    if ( grid_dim != 2 ) {
        throw std::runtime_error( "Only 2D-cartesian grids are supported, so far..." );
    }

    std::tuple<int, int> dims;
    std::get<0>( dims ) = param.getDefault( "nx", 3 );
    std::get<1>( dims ) = param.getDefault( "ny", 5 );

    int ghostWidth = param.getDefault( "ghost_width", 1 );

    init2D( dims, ghostWidth );
}

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

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputCellCollectionOfScalar(std::string name)
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

        v.resize( number_of_cells_and_ghost_cells, 0.0 );

        for( int j = 0; j < cartdims[1]; ++j ) {
            for( int i = 0; i < cartdims[0]; ++i ) {
                if ( beg == end ) {
                    OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
                }
                cellAt( i, j, v ) = *beg;
                beg++;
            }
        }
    } else { // Constant value
        const double value = param_.get<double>( name );
        v = inputCellScalarWithDefault( name, value );
    }

    return v;
}

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

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputCellScalarWithDefault(std::string /*name*/, double d)
{    
    CartesianCollectionOfScalar v( number_of_cells_and_ghost_cells, 0.0 );

    for( int j = 0; j < cartdims[1]; ++j ) {
        for( int i = 0; i < cartdims[0]; ++i ) {
            cellAt( i, j, v ) = d;
        }
    }

    return v;
}

equelle::CartesianGrid::CartesianCollectionOfScalar equelle::CartesianGrid::inputFaceScalarWithDefault(std::string name, double d )
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

double& equelle::CartesianGrid::cellAt( int i, int j, equelle::CartesianGrid::CartesianCollectionOfScalar &coll )
{
    int origin = ghost_width * cellStrides[1] + ghost_width * cellStrides[0];

    int index = origin + j*cellStrides[1] + i*cellStrides[0];
    return coll[ index ];
}

double &equelle::CartesianGrid::faceAt(int i, int j, const equelle::CartesianGrid::Face face, equelle::CartesianGrid::CartesianCollectionOfScalar &coll)
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

void equelle::CartesianGrid::dumpGrid(const equelle::CartesianGrid::CartesianCollectionOfScalar &grid, std::ostream &stream)
{
	int num_columns = cartdims[0] + 2*ghost_width;
    for( int j = 0; j < cartdims[1] + 2*ghost_width; ++j ) {
    	int row_offset  = j*cellStrides[1];
        std::copy_n( grid.begin() + row_offset, num_columns - 1, std::ostream_iterator<double>( stream, "," ) );
        stream << grid[row_offset + num_columns-1];
        stream << std::endl;
    }
}

