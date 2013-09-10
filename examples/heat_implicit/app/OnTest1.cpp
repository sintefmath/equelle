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

    // EQUELLE: all_cells : Collection Of Cells On AllCells() = AllCells()
    CollOfCells all_cells = er.allCells();

    // EQUELLE: boundary_cells : Collection Of Cells On BoundaryCells() = BoundaryCells()
    CollOfCells boundary_cells = er.boundaryCells();

    // EQUELLE: all_vol : Collection Of Scalar On all_cells = Volume(all_cells)
    CollOfScalars       all_vol   = er.norm( all_cells );
    CollOfScalarsOnColl all_vol_2 = CollOfScalarsOnColl( er.norm(all_cells), all_cells );

    // If something like this is the way to go, EquelleRuntimeCPU::volume() etc. should be modified instead of using the constructor with
    // two arguments above. (That was only done to avoid modifying the 'er' for now...)

    // EQUELLE: boundary_vol : Collection Of Scalar On boundary_cells = Volume(boundary_cells)
    CollOfScalars       boundary_vol   = er.norm( boundary_cells );
    CollOfScalarsOnColl boundary_vol_2 = CollOfScalarsOnColl( er.norm(boundary_cells), boundary_cells );

    // EQUELLE: vol_diff : Collection Of Scalar On AllCells() = all_vol - boundary_vol # NB! Should fail, preferably in the Equelle compiler
    CollOfScalars       vol_diff   = all_vol - boundary_vol;
    CollOfScalarsOnColl vol_diff_2 = all_vol_2 - boundary_vol_2;

    // EQUELLE: Output(vol_diff)
    er.output("vol_diff:       ",       vol_diff);
    er.output("all_vol_2:      ",      all_vol_2);
    er.output("boundary_vol_2: ", boundary_vol_2);
    er.output("vol_diff_2:     ",     vol_diff_2);

    return 0;
}







































