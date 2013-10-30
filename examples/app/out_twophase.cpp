
// This program was created by the Equelle compiler from SINTEF.

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

    // ============= Generated code starts here ================

    const CollOfScalar perm = er.userSpecifiedCollectionOfScalar("perm", er.allCells());
    const CollOfScalar poro = er.userSpecifiedCollectionOfScalar("poro", er.allCells());
    const Scalar watervisc = er.userSpecifiedScalarWithDefault("watervisc", double(0.0005));
    const Scalar oilvisc = er.userSpecifiedScalarWithDefault("oilvisc", double(0.005));
    const CollOfScalar vol = er.norm(er.allCells());
    auto computeTransmissibilities = [&](const CollOfScalar& permeability) -> CollOfScalar {
        const CollOfFace interior_faces = er.interiorFaces();
        const CollOfCell first = er.firstCell(interior_faces);
        const CollOfCell second = er.secondCell(interior_faces);
        const CollOfVector cdiff1 = (er.centroid(first) - er.centroid(interior_faces));
        const CollOfVector cdiff2 = (er.centroid(second) - er.centroid(interior_faces));
        const CollOfScalar p1 = er.operatorOn(permeability, er.allCells(), first);
        const CollOfScalar p2 = er.operatorOn(permeability, er.allCells(), second);
        const CollOfScalar a = er.norm(interior_faces);
        const CollOfScalar halftrans1 = ((a * p1) * (er.dot(er.normal(interior_faces), cdiff1) / er.dot(cdiff1, cdiff1)));
        const CollOfScalar halftrans2 = ((a * p2) * (er.dot(er.normal(interior_faces), cdiff2) / er.dot(cdiff2, cdiff2)));
        const CollOfScalar trans = (double(1) / ((double(1) / halftrans1) + (double(1) / halftrans2)));
        return trans;
    };
    const CollOfScalar trans = computeTransmissibilities(perm);
    auto upwind = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto computeTotalFluxes = [&](const CollOfScalar& pressure, const CollOfScalar& total_mobility) -> CollOfScalar {
        const CollOfScalar ngradp = -er.gradient(pressure);
        const CollOfScalar face_total_mobility = upwind(ngradp, total_mobility);
        return ((trans * face_total_mobility) * ngradp);
    };
    auto computePressureResidual = [&](const CollOfScalar& pressure, const CollOfScalar& total_mobility, const CollOfScalar& source) -> CollOfScalar {
        const CollOfScalar fluxes = computeTotalFluxes(pressure, total_mobility);
        return (er.divergence(fluxes) - source);
    };
    auto computeWaterMob = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeOilMob = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto computeTransportResidual = [&](const CollOfScalar& sw, const CollOfScalar& sw0, const CollOfScalar& fluxes, const CollOfScalar& source, const CollOfScalar& insource_sw, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar insource = er.trinaryIf((source > double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar outsource = er.trinaryIf((source < double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar mw = computeWaterMob(sw);
        const CollOfScalar mo = computeOilMob(sw);
        const CollOfScalar f = (mw / (mw + mo));
        const CollOfScalar f_face = upwind(fluxes, f);
        const CollOfScalar w_fluxes = (f_face * fluxes);
        const CollOfScalar q = ((insource * insource_sw) + (outsource * f));
        return (((sw - sw0) + ((dt / vol) * er.divergence(w_fluxes))) - q);
    };
    const SeqOfScalar timesteps = er.userSpecifiedSequenceOfScalar("timesteps");
    const CollOfScalar sw_initial = er.userSpecifiedCollectionOfScalar("sw_initial", er.allCells());
    const CollOfCell source_cells = er.userSpecifiedCollectionOfCellSubsetOf("source_cells", er.allCells());
    const CollOfScalar source_values = er.userSpecifiedCollectionOfScalar("source_values", source_cells);
    const CollOfScalar source = er.operatorExtend(source_values, source_cells, er.allCells());
    const CollOfScalar insource_sw = er.operatorExtend(double(1), er.allCells());
    CollOfScalar sw0;
    sw0 = sw_initial;
    CollOfScalar p0;
    p0 = er.operatorExtend(double(0), er.allCells());
    er.output("pressure", p0);
    er.output("saturation", sw0);
    for (const Scalar& dt : timesteps) {
        const CollOfScalar total_mobility = (computeWaterMob(sw0) + computeOilMob(sw0));
        auto pressureResLocal = [&](const CollOfScalar& pressure) -> CollOfScalar {
            return computePressureResidual(pressure, total_mobility, source);
        };
        const CollOfScalar p = er.newtonSolve(pressureResLocal, p0);
        const CollOfScalar fluxes = computeTotalFluxes(p, total_mobility);
        auto transportResLocal = [&](const CollOfScalar& sw) -> CollOfScalar {
            return computeTransportResidual(sw, sw0, fluxes, source, insource_sw, dt);
        };
        const CollOfScalar sw = er.newtonSolve(transportResLocal, sw0);
        p0 = p;
        sw0 = sw;
        er.output("pressure", p0);
        er.output("saturation", sw0);
    }

    // ============= Generated code ends here ================

    return 0;
}
