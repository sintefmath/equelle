/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>

#include "EquelleRuntimeCPU.hpp"




int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCPU er(param);

    // all_cells : Collection Of Cells On AllCells() = AllCells()
    CollOfCells all_cells = er.allCells();

    std::cout << "Total number of cells: " << all_cells.size() << std::endl;

    // boundary_cells : Collection Of Cells On BoundaryCells() = BoundaryCells()
    CollOfCells boundary_cells = er.boundaryCells();

    std::cout << "Number of boundary cells: " << boundary_cells.size() << std::endl;

    // all_vol : Collection Of Scalar On all_cells = Volume(all_cells)
    CollOfScalars all_vol = er.volume( all_cells );
    CollOfScalarsOnColl all_vol_2( er.volume(all_cells), all_cells );

    // boundary_vol : Collection Of Scalar On boundary_cells = Volume(boundary_cells)
    CollOfScalars boundary_vol = er.volume( boundary_cells );
    CollOfScalarsOnColl boundary_vol_2( er.volume(boundary_cells), boundary_cells );

    // vol_diff : Collection Of Scalar On AllCells() = all_vol - boundary_vol # NB! Should fail, preferably in the Equelle compiler
    CollOfScalars vol_diff = all_vol - boundary_vol;
    CollOfScalarsOnColl vol_diff_2 = all_vol_2 - boundary_vol_2;

    // Output(vol_diff)
    er.output("vol_diff:       ",       vol_diff);
    er.output("all_vol_2:      ",      all_vol_2);
    er.output("boundary_vol_2: ", boundary_vol_2);
    er.output("vol_diff_2:     ",     vol_diff_2);

    return 0;
}







































