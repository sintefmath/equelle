#include "equelle/SubGridBuilder.hpp"

#include <opm/core/grid.h>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <set>
#include <algorithm>
#include <iostream>
#include <iterator>

namespace equelle {
SubGrid SubGridBuilder::build(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract)
{
    SubGrid subGrid;
    subGrid.c_grid = allocate_grid( grid->dimensions, cellsToExtract.size(), 0, 0, 0, 0 );

    // Extract the inner-neighbors
    Opm::HelperOps helperOps( *grid );
    Opm::HelperOps::M adj = helperOps.div * helperOps.ngrad;

    std::cout << adj << std::endl;

    std::set<int> neighborCells;

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

    subGrid.global_cell = cellsToExtract;

    std::set_difference( neighborCells.begin(), neighborCells.end(), cellsToExtract.begin(), cellsToExtract.end(),
                         std::back_inserter( subGrid.global_cell ) );
/*  // Debug prints
    std::copy( neighborCells.begin(), neighborCells.end(), std::ostream_iterator<int>( std::cout, " " ) ); std::cout << std::endl;
    std::copy( subGrid.global_cell.begin(), subGrid.global_cell.end(), std::ostream_iterator<int>( std::cout, " " ) ); std::cout << std::endl;
*/
    subGrid.number_of_ghost_cells = subGrid.global_cell.size() - cellsToExtract.size();

    return subGrid;
}

SubGridBuilder::SubGridBuilder()
{
}

}
