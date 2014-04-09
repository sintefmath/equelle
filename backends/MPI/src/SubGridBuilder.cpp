#include "equelle/SubGridBuilder.hpp"

#include <opm/core/grid.h>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <iostream>
#include <iterator>

#include "equelle/mpiutils.hpp"
#include "equelle/ZoltanGrid.hpp"

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

SubGridBuilder::face_mapping
SubGridBuilder::extractNeighborFaces(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract )
{    
    std::unordered_map<int, int> old2new;

    // new_cell_facepos will be of size numCells + 1, so we make the first element zero.
    std::vector<int> new_cell_facepos( 1, 0 ); new_cell_facepos.reserve( cellsToExtract.size() );
    std::vector<int> new_cell_faces;

    for( int cell_index = 0; cell_index < cellsToExtract.size(); cell_index++ ) {
        const int cell = cellsToExtract[cell_index];
        const int startIndex = grid->cell_facepos[cell];
        const int endIndex   = grid->cell_facepos[cell+1];

        for( int i = startIndex; i < endIndex; ++i ) {
            const auto old_face_index = grid->cell_faces[i];

            // Create the new face index.
            if ( old2new.find( old_face_index ) == old2new.end() ) {
                const int new_face_index = old2new.size(); // Get the size before we start looking for the key!
                old2new[old_face_index] = new_face_index;
            }

            const auto new_face_index = old2new[old_face_index];
            new_cell_faces.push_back( new_face_index );
        }

        new_cell_facepos.push_back( endIndex - startIndex + new_cell_facepos.back() );
    }

    // Invert the list of indices.
    std::vector<int> global_face( old2new.size(), -1 );

    for( auto it: old2new ) {
        global_face[it.second] =  it.first;
    }

    // Set up the return structure.
    face_mapping fmap;
    fmap.cell_facepos = new_cell_facepos;
    fmap.cell_faces   = new_cell_faces;
    fmap.global_face  = global_face;
    fmap.cell_global_to_local = old2new;

    return fmap;
}

SubGridBuilder::node_mapping
SubGridBuilder::extractNeighborNodes(const UnstructuredGrid *grid, const std::vector<int> &globalFaces )
{
    std::unordered_map<int, int> old2new;
    node_mapping nm;
    nm.face_nodepos.reserve( globalFaces.size() );
    nm.face_nodepos.push_back( 0 );

    for( auto face: globalFaces ) {
        const int startIndex = grid->face_nodepos[face];
        const int endIndex   = grid->face_nodepos[face+1];

        for( auto i = startIndex; i < endIndex; ++i ) {
            const auto old_node_index = grid->face_nodes[i];

            if ( old2new.find( old_node_index ) == old2new.end() ) {
                const int new_node_index = old2new.size();
                old2new[old_node_index] = new_node_index;
            }

            const auto new_node_index = old2new[old_node_index];
            nm.face_nodes.push_back( new_node_index );
        }

        nm.face_nodepos.push_back( endIndex - startIndex + nm.face_nodepos.back() );
    }

    // Invert the list of indices.
    nm.global_node.resize( old2new.size(), -1 );

    for( auto it: old2new ) {
        nm.global_node[it.second] = it.first;
    }
    nm.face_global_to_local = old2new;
    return nm;
}


template<typename T>
void reduceAndReindex( const T* src, T* dst, const int* oldIndices, const int numNew, const int dim = 1 ) {
    for( int newIdx = 0; newIdx < numNew; ++newIdx ) {
        int oldIdx = oldIndices[newIdx];
        std::copy_n( &(src[dim*oldIdx]), dim, &(dst[dim*newIdx]) );
    }
}

void SubGridBuilder::build_face_cells( const face_mapping &participatingFaces,
                                       SubGrid &subGrid, const UnstructuredGrid* grid)
{
    std::unordered_map<int, int> cell_glob2loc;
    for( int i = 0; i < subGrid.global_cell.size(); ++i ) {
        cell_glob2loc[ subGrid.global_cell[i] ] = i;
    }

    for( int lface = 0; lface < participatingFaces.global_face.size(); ++lface ) {
        int gface = participatingFaces.global_face[lface];

        // A face always point to 2 cells, which might be boundary cells.
        for( int i = 0; i < 2; ++i ) {
            const int gcell = grid->face_cells[2*gface + i];

            int lcell;

            if ( gcell == Boundary::outer ) {
                lcell = Boundary::outer;
            } else { // Check if the cells is part of the subgrid or an inner-boundary.
                auto it = cell_glob2loc.find( gcell );
                lcell = ( it != cell_glob2loc.end() ) ? it->second : Boundary::inner;
            }

            subGrid.c_grid->face_cells[2*lface + i] = lcell;
        }
    }
}

SubGrid SubGridBuilder::build(const UnstructuredGrid* grid, const std::vector<int>& cellsToExtract )
{
    SubGrid subGrid;

    // Extract the cells and ghost-cells that that will be part of our subdomain    
    std::set<int> neighborCells = extractNeighborCells(grid, cellsToExtract);

    // Build up the local to global mapping based on the input and the additional neighbor cells found above
    subGrid.global_cell = cellsToExtract;
    std::set_difference( neighborCells.begin(), neighborCells.end(), cellsToExtract.begin(), cellsToExtract.end(),
                         std::back_inserter( subGrid.global_cell ) );

    subGrid.number_of_ghost_cells = subGrid.global_cell.size() - cellsToExtract.size();

    auto participatingFaces = extractNeighborFaces(grid, subGrid.global_cell);
    auto participatingNodes = extractNeighborNodes(grid, participatingFaces.global_face);

    subGrid.global_face = participatingFaces.global_face;
    subGrid.face_global_to_local = participatingFaces.cell_global_to_local;

    subGrid.c_grid = allocate_grid( grid->dimensions, subGrid.global_cell.size(),
                                    participatingFaces.global_face.size(), participatingNodes.face_nodes.size(),
                                    participatingFaces.cell_faces.size(), participatingNodes.global_node.size() );

    // Copy the face information into the subGrid
    std::copy( begin( participatingFaces.cell_facepos ), end( participatingFaces.cell_facepos ),
               subGrid.c_grid->cell_facepos );
    std::copy( begin( participatingFaces.cell_faces ), end( participatingFaces.cell_faces ),
               subGrid.c_grid->cell_faces );

    // Copy the node information into the subGrid
    std::copy( begin( participatingNodes.face_nodepos ), end( participatingNodes.face_nodepos ),
               subGrid.c_grid->face_nodepos );
    std::copy( begin( participatingNodes.face_nodes ), end( participatingNodes.face_nodes ),
               subGrid.c_grid->face_nodes );

    const int dim = grid->dimensions;

    build_face_cells( participatingFaces, subGrid, grid );

    // Reindex for addressing based on cells
    reduceAndReindex( grid->cell_centroids, subGrid.c_grid->cell_centroids, subGrid.global_cell.data(), subGrid.global_cell.size(), dim );
    reduceAndReindex( grid->cell_volumes, subGrid.c_grid->cell_volumes, subGrid.global_cell.data(), subGrid.global_cell.size() );

    // Reindex for addressing based on faces
    auto& global_face = participatingFaces.global_face;
    reduceAndReindex( grid->face_areas, subGrid.c_grid->face_areas, global_face.data(), global_face.size() );
    reduceAndReindex( grid->face_centroids, subGrid.c_grid->face_centroids, global_face.data(), global_face.size(), dim );
    reduceAndReindex( grid->face_normals, subGrid.c_grid->face_normals, global_face.data(), global_face.size(), dim );

    // Reindex for addressing based on nodes
    auto& global_node = participatingNodes.global_node;
    reduceAndReindex( grid->node_coordinates, subGrid.c_grid->node_coordinates, global_node.data(), global_node.size(), dim );

    return subGrid;
}

SubGridBuilder::SubGridBuilder()
{
}

int GridQuerying::numFaces(const UnstructuredGrid *grid, int cell)
{
    return grid->cell_facepos[cell+1] - grid->cell_facepos[cell];
}

int GridQuerying::numNodes(const UnstructuredGrid *grid, int face)
{
    return grid->face_nodepos[face+1] - grid->face_nodepos[face];
}

CollOfCell SubGrid::map_to_global(const CollOfCell &local_collection)
{
    CollOfCell global_collection;

    for( auto x: local_collection ) {
        global_collection.emplace_back( global_cell[ x.index ] );
    }

    return global_collection;
}

CollOfFace SubGrid::map_to_global(const CollOfFace &local_collection)
{
    CollOfFace global_collection;
    for( auto x: local_collection ) {
        global_collection.emplace_back( global_face[x.index] );
    }

    return global_collection;
}


SubGrid::~SubGrid()
{
    //destroy_grid( c_grid );
}

}
