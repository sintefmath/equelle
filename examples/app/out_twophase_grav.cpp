
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

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er);
void ensureRequirements(const equelle::EquelleRuntimeCPU& er);

#ifndef EQUELLE_NO_MAIN
int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    equelle::EquelleRuntimeCPU er(param);
    equelleGeneratedCode(er);
    return 0;
}
#endif // EQUELLE_NO_MAIN

void equelleGeneratedCode(equelle::EquelleRuntimeCPU& er) {
    using namespace equelle;
    ensureRequirements(er);

    // ============= Generated code starts here ================

    const CollOfScalar perm = er.inputCollectionOfScalar("perm", er.allCells());
    const CollOfScalar poro = er.inputCollectionOfScalar("poro", er.allCells());
    const Scalar watervisc = er.inputScalarWithDefault("watervisc", double(0.0005));
    const Scalar oilvisc = er.inputScalarWithDefault("oilvisc", double(0.005));
    const Scalar waterdensity = er.inputScalarWithDefault("waterdensity", double(1000));
    const Scalar oildensity = er.inputScalarWithDefault("oildensity", double(750));
    const Scalar gravity = er.inputScalarWithDefault("gravity", double(9.82));
    const CollOfScalar pv = (poro * er.norm(er.allCells()));
    const CollOfScalar cell_depths = CollOfScalar(er.centroid(er.allCells()).col(1));
    const CollOfScalar zdiff = er.gradient(cell_depths);
    auto computeTransmissibilities = [&](const CollOfScalar& permeability) -> CollOfScalar {
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
    const CollOfScalar trans = computeTransmissibilities(perm);
    auto upwind_i2_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto upwind_i4_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto upwind_i9_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto upwind_i11_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto upwind_i15_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        return er.trinaryIf((flux >= double(0)), x1, x2);
    };
    auto computeWaterMob_i1_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeWaterMob_i8_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeWaterMob_i13_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeOilMob_i3_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto computeOilMob_i10_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto computeOilMob_i14_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto fluxWithGrav_i5_ = [&](const CollOfScalar& pressure, const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar ngradp = -er.gradient(pressure);
        er.output("ngradp", ngradp);
        const CollOfScalar flux_w = (ngradp + ((gravity * waterdensity) * zdiff));
        const CollOfScalar flux_o = (ngradp + ((gravity * oildensity) * zdiff));
        er.output("flux_o", flux_o);
        er.output("swinflux", sw);
        const CollOfScalar face_mob_w = upwind_i9_(flux_w, computeWaterMob_i8_(sw));
        const CollOfScalar face_mob_o = upwind_i11_(flux_o, computeOilMob_i10_(sw));
        er.output("face_mob_o", face_mob_o);
        const CollOfScalar face_total_mobility = (face_mob_w + face_mob_o);
        er.output("ftm", face_total_mobility);
        const CollOfScalar omega = (((face_mob_w * waterdensity) + (face_mob_o * oildensity)) / face_total_mobility);
        return ((trans * face_total_mobility) * (ngradp + ((gravity * omega) * zdiff)));
    };
    auto fluxWithGrav_i12_ = [&](const CollOfScalar& pressure, const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar ngradp = -er.gradient(pressure);
        er.output("ngradp", ngradp);
        const CollOfScalar flux_w = (ngradp + ((gravity * waterdensity) * zdiff));
        const CollOfScalar flux_o = (ngradp + ((gravity * oildensity) * zdiff));
        er.output("flux_o", flux_o);
        er.output("swinflux", sw);
        const CollOfScalar face_mob_w = upwind_i9_(flux_w, computeWaterMob_i8_(sw));
        const CollOfScalar face_mob_o = upwind_i11_(flux_o, computeOilMob_i10_(sw));
        er.output("face_mob_o", face_mob_o);
        const CollOfScalar face_total_mobility = (face_mob_w + face_mob_o);
        er.output("ftm", face_total_mobility);
        const CollOfScalar omega = (((face_mob_w * waterdensity) + (face_mob_o * oildensity)) / face_total_mobility);
        return ((trans * face_total_mobility) * (ngradp + ((gravity * omega) * zdiff)));
    };
    auto computePressureResidual = [&](const CollOfScalar& pressure, const CollOfScalar& sw, const CollOfScalar& source) -> CollOfScalar {
        const CollOfScalar flux = fluxWithGrav_i5_(pressure, sw);
        er.output("fluxinres", flux);
        return (er.divergence(flux) - source);
    };
    auto computeTransportResidual = [&](const CollOfScalar& sw, const CollOfScalar& sw0, const CollOfScalar& flux, const CollOfScalar& source, const CollOfScalar& insource_sw, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar insource = er.trinaryIf((source > double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar outsource = er.trinaryIf((source < double(0)), source, er.operatorExtend(double(0), er.allCells()));
        const CollOfScalar mw = computeWaterMob_i13_(sw);
        const CollOfScalar mo = computeOilMob_i14_(sw);
        const CollOfScalar fracflow = (mw / (mw + mo));
        const CollOfScalar face_fracflow = upwind_i15_(flux, fracflow);
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
    auto sw0 = sw_initial;
    auto p0 = er.operatorExtend(double(0), er.allCells());
    er.output("pressure", p0);
    er.output("saturation", sw0);
    for (const Scalar& dt : timesteps) {
        auto pressureResLocal = [&](const CollOfScalar& pressure) -> CollOfScalar {
            return computePressureResidual(pressure, sw0, source);
        };
        const CollOfScalar p = er.newtonSolve(pressureResLocal, p0);
        const CollOfScalar flux = fluxWithGrav_i12_(p, sw0);
        auto transportResLocal = [&](const CollOfScalar& sw) -> CollOfScalar {
            return computeTransportResidual(sw, sw0, flux, source, insource_sw, dt);
        };
        const CollOfScalar sw = er.newtonSolve(transportResLocal, er.operatorExtend(double(0.5), er.allCells()));
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
