#include "equelle/SubGridBuilder.hpp"

#include <opm/core/grid.h>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <set>
#include <algorithm>
#include <iostream>
#include <iterator>

namespace equelle {

std::set<int> SubGridBuilder::extractNeighborCells(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract)
{
    std::set<int> neighborCells;

    // Extract the inner-neighbors

    Opm::HelperOps helperOps( *grid );
    Opm::HelperOps::M adj = helperOps.div * helperOps.ngrad;

    //std::cout << helperOps.div << std::endl;
    //std::cout << adj << std::endl;

    for( int i = 0; i < cellsToExtract.size(); ++i ) {
        int cell = cellsToExtract[i];
        for( Eigen::SparseMatrix<double>::InnerIterator it( adj, cell); it; ++it ) {
            if( it.row() == it.col() ) { // Skip diagonal elements
                // NOP
            } else {
                // By using it.index() we can ignore what is the inner and outer dimensions. (Ie col-major or row-major.)
                neighborCells.insert( it.index() );
            }
        }
    }

    return neighborCells;
}

std::set<int> SubGridBuilder::extractNeighborFaces(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract)
{
    Opm::HelperOps helperOps( *grid );
    std::set<int> participatingFaces;

    std::cout << helperOps.ngrad << std::endl;

    for( int i = 0; i < cellsToExtract.size(); ++i ) {
        int gid = cellsToExtract[i];
        for( Eigen::SparseMatrix<double>::InnerIterator it( helperOps.ngrad, gid ); it; ++it ) {
            //std::cout << it.row() << ", " << it.col() << std::endl;
            participatingFaces.insert( it.row() );

        }
    }

    return participatingFaces;
}

SubGrid SubGridBuilder::build(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract)
{
    SubGrid subGrid;

    std::set<int> neighborCells = extractNeighborCells(grid, cellsToExtract);

    subGrid.global_cell = cellsToExtract;

    std::set_difference( neighborCells.begin(), neighborCells.end(), cellsToExtract.begin(), cellsToExtract.end(),
                         std::back_inserter( subGrid.global_cell ) );
    /*  // Debug prints
    std::copy( neighborCells.begin(), neighborCells.end(), std::ostream_iterator<int>( std::cout, " " ) ); std::cout << std::endl;
    std::copy( subGrid.global_cell.begin(), subGrid.global_cell.end(), std::ostream_iterator<int>( std::cout, " " ) ); std::cout << std::endl;
*/
    subGrid.number_of_ghost_cells = subGrid.global_cell.size() - cellsToExtract.size();

    subGrid.c_grid = allocate_grid( grid->dimensions, subGrid.global_cell.size(), 0, 0, 0, 0 );

    // We now have the new indexing for cells, so we extract all cell data we can based on that indexing
    const int dim = grid->dimensions;
    for( int lid = 0; lid < subGrid.global_cell.size(); ++lid ) {
        int gid = subGrid.global_cell[lid];

        std::copy_n( &(grid->cell_centroids[dim*gid]), dim, &(subGrid.c_grid->cell_centroids[dim*lid]) );

        subGrid.c_grid->cell_volumes[lid] = grid->cell_volumes[gid];
    }

    // We are now ready to extract all the faces participating in our subdomain
    auto participatingFaces = extractNeighborFaces(grid, subGrid.global_cell);
    std::copy( participatingFaces.begin(), participatingFaces.end(), std::ostream_iterator<int>( std::cout, " " ) );
    std::cout << std::endl;



    return subGrid;
}

SubGridBuilder::SubGridBuilder()
{
}

}
