#pragma once

#include <vector>
#include <set>

class UnstructuredGrid;


namespace equelle {

/** UnstructuredSubGrid is as an augmented Opm::UnstructuredGrid with */
struct SubGrid {
    UnstructuredGrid* c_grid;

    int number_of_ghost_cells;

    /** Store the global cell index of each cell in SubGrid. */
    std::vector<int> global_cell;
};

/** SubGridBuilder is responsible for building a subgrid of an Opm::UnstructuredGrid,
 *  given the export list from Zoltan.
 */
class SubGridBuilder
{
public:       
    static SubGrid build( const UnstructuredGrid* grid, const std::vector<int>& cellsToExtract );

private:
    SubGridBuilder();

    struct face_mapping {
        std::vector<int> cell_facepos; //! Mirrors UnstructuredGrid::cell_facepos.
        std::vector<int> cell_faces;   //! Mirrors UnstructuredGrid::cell_faces.
        std::vector<int> global_face;  //! The global face index of each face in the subgrid. */
    };

    static std::set<int> extractNeighborCells(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract);
    static face_mapping extractNeighborFaces(const UnstructuredGrid *grid, const std::vector<int> &cellsToExtract);
};


struct GridQuerying {
    /** Return the number of faces for a cell. */
    static int numFaces( const UnstructuredGrid* grid, int cell );
};

} // namespace equelle



