
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
#include <array>

#include "equelle/EquelleRuntimeCPU.hpp"
#include "equelle/CartesianGrid.hpp"//Should be renamed EquelleCartesianRuntimeCPU

void ensureRequirements(const equelle::EquelleRuntimeCPU& er);
void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er, equelle::CartesianEquelleRuntime& er_cart);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::CartesianEquelleRuntime er_cart(param);
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er, er_cart);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er,
                          equelle::CartesianEquelleRuntime& er_cart) {
    using namespace equelle;
    ensureRequirements(er);
    (void)er_cart; // To suppress compile warnings if not used below.

    // ============= Generated code starts here ================

    const CollOfScalar perm = er.inputCollectionOfScalar("perm", er.allCells());
    const CollOfScalar poro = er.inputCollectionOfScalar("poro", er.allCells());
    const Scalar watervisc = er.inputScalarWithDefault("watervisc", double(0.0005));
    const Scalar oilvisc = er.inputScalarWithDefault("oilvisc", double(0.005));
    const CollOfScalar pv = (poro * er.norm(er.allCells()));
    auto computeTrans = [&](const auto& permeability) {
        const CollOfFace interior_faces = er.interiorFaces();
        const CollOfCell first = er.firstCell(interior_faces);
        const CollOfCell second = er.secondCell(interior_faces);
        const CollOfVector cdiff1 = (er.centroid(first) - er.centroid(interior_faces));
        const CollOfVector cdiff2 = (er.centroid(second) - er.centroid(interior_faces));
        const CollOfScalar p1 = er.operatorOn(permeability, er.allCells(), first);
        const CollOfScalar p2 = er.operatorOn(permeability, er.allCells(), second);
        const CollOfScalar a = er.norm(interior_faces);
        const CollOfScalar halftrans1 = ((-a * p1) * (er.dot(er.normal(interior_faces), cdiff1) / er.dot(cdiff1, cdiff1)));
        const CollOfScalar halftrans2 = ((a * p2) * (er.dot(er.normal(interior_faces), cdiff2) / er.dot(cdiff2, cdiff2)));
        const CollOfScalar trans = (double(1) / ((double(1) / halftrans1) + (double(1) / halftrans2)));
        return trans;
    };
    const CollOfScalar trans = computeTrans(perm);
    auto upwind = [&](const auto& flux, const auto& x) {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto computeTotalFlux = [&](const auto& pressure, const auto& total_mob) {
        const CollOfScalar ngradp = -er.gradient(pressure);
        const CollOfScalar face_total_mobility = upwind(ngradp, total_mob);
        return ((trans * face_total_mobility) * ngradp);
    };
    auto computePressureResidual = [&](const auto& pressure, const auto& total_mob, const auto& source) {
        const CollOfScalar flux = computeTotalFlux(pressure, total_mob);
        return (er.divergence(flux) - source);
    };
    auto computeWaterMob = [&](const auto& sw) {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeOilMob = [&](const auto& sw) {
        const CollOfScalar so = (er.operatorExtend(double(1), er.allCells()) - sw);
        const CollOfScalar kro = so;
        return (kro / oilvisc);
    };
    auto computeTransportResidual = [&](const auto& sw, const auto& sw0, const auto& flux, const auto& source, const auto& insource_sw, const auto& dt) {
        const CollOfScalar insource = er.trinaryIf((source > double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar outsource = er.trinaryIf((source < double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar mw = computeWaterMob(sw);
        const CollOfScalar mo = computeOilMob(sw);
        const CollOfScalar fracflow = (mw / (mw + mo));
        const CollOfScalar face_fracflow = upwind(flux, fracflow);
        const CollOfScalar water_flux = (face_fracflow * flux);
        const CollOfScalar q = ((insource * insource_sw) + (outsource * fracflow));
        return ((sw - sw0) + ((dt / pv) * (er.divergence(water_flux) - q)));
    };
    const SeqOfScalar timesteps = er.inputSequenceOfScalar("timesteps");
    const CollOfScalar sw_initial = er.inputCollectionOfScalar("sw_initial", er.allCells());
    const CollOfCell source_cells = er.inputDomainSubsetOf("source_cells", er.allCells());
    const CollOfScalar source_values = er.inputCollectionOfScalar("source_values", source_cells);
    const CollOfScalar source = er.operatorExtend(source_values, source_cells, er.allCells());
    const CollOfScalar insource_sw = er.operatorExtend(double(1), er.allCells());
    CollOfScalar sw0 = sw_initial;
    CollOfScalar p0 = er.operatorExtend(double(0), er.allCells());
    er.output("pressure", p0);
    er.output("saturation", sw0);
    for (const Scalar& dt : timesteps) {
        auto pressureResLocal = [&](const auto& p, const auto& sw) {
            const CollOfScalar total_mobility = (computeWaterMob(sw) + computeOilMob(sw));
            return computePressureResidual(p, total_mobility, source);
        };
        auto transportResLocal = [&](const auto& p, const auto& sw) {
            const CollOfScalar total_mobility = (computeWaterMob(sw) + computeOilMob(sw));
            const CollOfScalar flux = computeTotalFlux(p, total_mobility);
            return computeTransportResidual(sw, sw0, flux, source, insource_sw, dt);
        };
        const auto newvals = er.newtonSolveSystem(makeArray(pressureResLocal, transportResLocal), makeArray(p0, er.operatorExtend(double(0.5), er.allCells())));
        p0 = std::get<0>(newvals);
        sw0 = std::get<1>(newvals);
        er.output("pressure", p0);
        er.output("saturation", sw0);
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
