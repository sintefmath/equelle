
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

    const auto perm = er.inputCollectionOfScalar("perm", er.allCells());
    const auto poro = er.inputCollectionOfScalar("poro", er.allCells());
    const auto watervisc = er.inputScalarWithDefault("watervisc", double(0.0005));
    const auto oilvisc = er.inputScalarWithDefault("oilvisc", double(0.005));
    const auto waterdensity = er.inputScalarWithDefault("waterdensity", double(1000));
    const auto oildensity = er.inputScalarWithDefault("oildensity", double(750));
    const auto gravity = er.inputScalarWithDefault("gravity", double(9.82));
    const auto pv = (poro * er.norm(er.allCells()));
    const auto cell_depths = CollOfScalar(er.centroid(er.allCells()).col(1));
    const auto zdiff = er.gradient(cell_depths);
    auto computeTransmissibilities = [&](const auto& permeability) {
        const auto interior_faces = er.interiorFaces();
        const auto first = er.firstCell(interior_faces);
        const auto second = er.secondCell(interior_faces);
        const auto cdiff1 = (er.centroid(first) - er.centroid(interior_faces));
        const auto cdiff2 = (er.centroid(second) - er.centroid(interior_faces));
        const auto p1 = er.operatorOn(permeability, er.allCells(), first);
        const auto p2 = er.operatorOn(permeability, er.allCells(), second);
        const auto a = er.norm(interior_faces);
        const auto halftrans1 = ((-a * p1) * (er.dot(er.normal(interior_faces), cdiff1) / er.dot(cdiff1, cdiff1)));
        const auto halftrans2 = ((a * p2) * (er.dot(er.normal(interior_faces), cdiff2) / er.dot(cdiff2, cdiff2)));
        const auto trans = (double(1) / ((double(1) / halftrans1) + (double(1) / halftrans2)));
        return trans;
    };
    const auto trans = computeTransmissibilities(perm);
    auto upwind = [&](const auto& flux, const auto& x) {
        const auto x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const auto x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto computeWaterMob = [&](const auto& sw) {
        const auto krw = sw;
        return (krw / watervisc);
    };
    auto computeOilMob = [&](const auto& sw) {
        const auto kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto fluxWithGrav = [&](const auto& pressure, const auto& sw) {
        const auto ngradp = -er.gradient(pressure);
        er.output("ngradp", ngradp);
        const auto flux_w = (ngradp + ((gravity * waterdensity) * zdiff));
        const auto flux_o = (ngradp + ((gravity * oildensity) * zdiff));
        er.output("flux_o", flux_o);
        er.output("swinflux", sw);
        const auto face_mob_w = upwind(flux_w, computeWaterMob(sw));
        const auto face_mob_o = upwind(flux_o, computeOilMob(sw));
        er.output("face_mob_o", face_mob_o);
        const auto face_total_mobility = (face_mob_w + face_mob_o);
        er.output("ftm", face_total_mobility);
        const auto omega = (((face_mob_w * waterdensity) + (face_mob_o * oildensity)) / face_total_mobility);
        return ((trans * face_total_mobility) * (ngradp + ((gravity * omega) * zdiff)));
    };
    auto computePressureResidual = [&](const auto& pressure, const auto& sw, const auto& source) {
        const auto flux = fluxWithGrav(pressure, sw);
        er.output("fluxinres", flux);
        return (er.divergence(flux) - source);
    };
    auto computeTransportResidual = [&](const auto& sw, const auto& sw0, const auto& flux, const auto& source, const auto& insource_sw, const auto& dt) {
        const auto insource = er.trinaryIf((source > double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const auto outsource = er.trinaryIf((source < double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const auto mw = computeWaterMob(sw);
        const auto mo = computeOilMob(sw);
        const auto fracflow = (mw / (mw + mo));
        const auto face_fracflow = upwind(flux, fracflow);
        const auto water_flux = (face_fracflow * flux);
        const auto q = ((insource * insource_sw) + (outsource * fracflow));
        return ((sw - sw0) + ((dt / pv) * (er.divergence(water_flux) - q)));
    };
    const auto timesteps = er.inputSequenceOfScalar("timesteps");
    const auto sw_initial = er.inputCollectionOfScalar("sw_initial", er.allCells());
    const auto source_cells = er.inputDomainSubsetOf("source_cells", er.allCells());
    const auto source_values = er.inputCollectionOfScalar("source_values", source_cells);
    const auto source = er.operatorExtend(source_values, source_cells, er.allCells());
    const auto insource_sw = er.operatorExtend(double(1), er.allCells());
    auto sw0 = sw_initial;
    auto p0 = er.operatorExtend(double(0), er.allCells());
    er.output("pressure", p0);
    er.output("saturation", sw0);
    for (const Scalar& dt : timesteps) {
        auto pressureResLocal = [&](const auto& pressure) {
            return computePressureResidual(pressure, sw0, source);
        };
        const auto p = er.newtonSolve(pressureResLocal, p0);
        const auto flux = fluxWithGrav(p, sw0);
        auto transportResLocal = [&](const auto& sw) {
            return computeTransportResidual(sw, sw0, flux, source, insource_sw, dt);
        };
        const auto sw = er.newtonSolve(transportResLocal, er.operatorExtend(double(0.5), er.allCells()));
        p0 = p;
        sw0 = sw;
        er.output("pressure", p0);
        er.output("flux", flux);
        er.output("saturation", sw0);
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    er.ensureGridDimensionMin(2);
}
