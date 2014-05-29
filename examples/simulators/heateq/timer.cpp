
// This program was created by the Equelle compiler from SINTEF.

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <array>
#include <ctime>

#include "EquelleRuntimeCUDA.hpp"

void ensureRequirements(const EquelleRuntimeCUDA& er);
void equelleGeneratedCode(equelleCUDA::EquelleRuntimeCUDA& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelleCUDA::EquelleRuntimeCUDA er(param);
    equelleGeneratedCode(er);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelleCUDA::EquelleRuntimeCUDA& er) {
    using namespace equelleCUDA;
    ensureRequirements(er);

    // ============= Generated code starts here ================

    const Scalar k = er.inputScalarWithDefault("k", double(0.3));
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    const CollOfScalar u0 = er.inputCollectionOfScalar("u_initial", er.allCells());
    const CollOfFace dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces());
    const CollOfScalar dirichlet_val = er.inputCollectionOfScalar("dirichlet_val", dirichlet_boundary);
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfFace interior_faces = er.interiorFaces();
    const CollOfCell first = er.firstCell(interior_faces);
    const CollOfCell second = er.secondCell(interior_faces);
    const CollOfScalar itrans = (k * (er.norm(interior_faces) / er.norm((er.centroid(first) - er.centroid(second)))));
    const CollOfFace bf = er.boundaryFaces();
    const CollOfCell bf_cells = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.secondCell(bf), er.firstCell(bf));
    const CollOfScalar bf_sign = er.trinaryIf(er.isEmpty(er.firstCell(bf)), er.operatorExtend(-double(1), bf), er.operatorExtend(double(1), bf));
    const CollOfScalar btrans = (k * (er.norm(bf) / er.norm((er.centroid(bf) - er.centroid(bf_cells)))));
    const CollOfScalar dir_sign = er.operatorOn(bf_sign, er.boundaryFaces(), dirichlet_boundary);
    std::function<CollOfScalar(const CollOfScalar&)> computeInteriorFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        return (-itrans * er.gradient(u));
    };
    std::function<CollOfScalar(const CollOfScalar&)> computeBoundaryFlux = [&](const CollOfScalar& u) -> CollOfScalar {
        const CollOfScalar u_dirbdycells = er.operatorOn(u, er.allCells(), er.operatorOn(bf_cells, er.boundaryFaces(), dirichlet_boundary));
        const CollOfScalar dir_fluxes = ((er.operatorOn(btrans, er.boundaryFaces(), dirichlet_boundary) * dir_sign) * (u_dirbdycells - dirichlet_val));
        return er.operatorExtend(dir_fluxes, dirichlet_boundary, er.boundaryFaces());
    };
    
    // Create timer
    std::clock_t start;
    double duration;
    int iteration_count = 0;
    
    CollOfScalar expU;
    expU = u0;
    for (const Scalar& dt : timesteps) {
	if ( iteration_count == 10) {
	    start = std::clock();
	}
        const CollOfScalar ifluxes = computeInteriorFlux(expU);
        const CollOfScalar bfluxes = computeBoundaryFlux(expU);
        const CollOfScalar fluxes = (er.operatorExtend(ifluxes, er.interiorFaces(), er.allFaces()) + er.operatorExtend(bfluxes, er.boundaryFaces(), er.allFaces()));
        expU = (expU - ((dt / vol) * er.divergence(fluxes)));
        //er.output("expU", expU);
        er.output("maximum of u", er.maxReduce(expU));
	
	iteration_count++;
	if ( iteration_count == 110 ) {
	    duration = (std::clock() - start) / (double) CLOCKS_PER_SEC;
	}
    }

    std::cout << "100 iterations took " << duration << " seconds\n";
    //    er.output("expU", expU);

    // ============= Generated code ends here ================

}

void ensureRequirements(const EquelleRuntimeCUDA& er)
{
    (void)er;
}
