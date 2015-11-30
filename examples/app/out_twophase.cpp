
// This program was created by the Equelle compiler from SINTEF.

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

    const CollOfScalar perm = (er.inputCollectionOfScalar("perm", er.allCells()) * double(1));
    const CollOfScalar poro_in = er.inputCollectionOfScalar("poro", er.allCells());
    const Scalar watervisc = (er.inputScalarWithDefault("watervisc", double(0.0005)) * double(1));
    const Scalar oilvisc = (er.inputScalarWithDefault("oilvisc", double(0.005)) * double(1));
    const CollOfScalar min_poro = er.operatorExtend(er.inputScalarWithDefault("min_poro", double(0.0001)), er.allCells());
    const CollOfScalar poro = er.trinaryIf((poro_in < min_poro), min_poro, poro_in);
    const CollOfScalar pv = (poro * er.norm(er.allCells()));
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
    auto upwind_i3_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        const CollOfScalar zero = (double(0) * flux);
        return er.trinaryIf((flux >= zero), x1, x2);
    };
    auto upwind_i7_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        const CollOfScalar zero = (double(0) * flux);
        return er.trinaryIf((flux >= zero), x1, x2);
    };
    auto upwind_i11_ = [&](const CollOfScalar& flux, const CollOfScalar& x) -> CollOfScalar {
        const CollOfScalar x1 = er.operatorOn(x, er.allCells(), er.firstCell(er.interiorFaces()));
        const CollOfScalar x2 = er.operatorOn(x, er.allCells(), er.secondCell(er.interiorFaces()));
        const CollOfScalar zero = (double(0) * flux);
        return er.trinaryIf((flux >= zero), x1, x2);
    };
    auto computeTotalFlux_i4_ = [&](const CollOfScalar& pressure, const CollOfScalar& total_mobility) -> CollOfScalar {
        const CollOfScalar ngradp = -er.gradient(pressure);
        const CollOfScalar face_total_mobility = upwind_i7_(ngradp, total_mobility);
        return ((trans * face_total_mobility) * ngradp);
    };
    auto computeTotalFlux_i8_ = [&](const CollOfScalar& pressure, const CollOfScalar& total_mobility) -> CollOfScalar {
        const CollOfScalar ngradp = -er.gradient(pressure);
        const CollOfScalar face_total_mobility = upwind_i7_(ngradp, total_mobility);
        return ((trans * face_total_mobility) * ngradp);
    };
    auto computePressureResidual = [&](const CollOfScalar& pressure, const CollOfScalar& total_mobility, const CollOfScalar& source) -> CollOfScalar {
        const CollOfScalar flux = computeTotalFlux_i4_(pressure, total_mobility);
        return (er.divergence(flux) - source);
    };
    auto computeWaterMob_i1_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeWaterMob_i9_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar krw = sw;
        return (krw / watervisc);
    };
    auto computeOilMob_i2_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto computeOilMob_i10_ = [&](const CollOfScalar& sw) -> CollOfScalar {
        const CollOfScalar kro = (er.operatorExtend(double(1), er.allCells()) - sw);
        return (kro / oilvisc);
    };
    auto computeTransportResidual = [&](const CollOfScalar& sw, const CollOfScalar& sw0, const CollOfScalar& flux, const CollOfScalar& source, const CollOfScalar& insource_sw, const Scalar& dt) -> CollOfScalar {
        const Scalar zero = double(0);
        const CollOfScalar insource = er.trinaryIf((source > zero), source, er.operatorExtend(zero, er.allCells()));
        const CollOfScalar outsource = er.trinaryIf((source < zero), source, er.operatorExtend(zero, er.allCells()));
        const CollOfScalar mw = computeWaterMob_i9_(sw);
        const CollOfScalar mo = computeOilMob_i10_(sw);
        const CollOfScalar fracflow = (mw / (mw + mo));
        const CollOfScalar face_fracflow = upwind_i11_(flux, fracflow);
        const CollOfScalar water_flux = (face_fracflow * flux);
        const CollOfScalar q = ((insource * insource_sw) + (outsource * fracflow));
        return ((sw - sw0) + ((dt / pv) * (er.divergence(water_flux) - q)));
    };
    const SeqOfScalar timesteps = (er.inputSequenceOfScalar("timesteps") * double(1));
    const CollOfScalar sw_initial = er.inputCollectionOfScalar("sw_initial", er.allCells());
    const CollOfCell source_cells = er.inputDomainSubsetOf("source_cells", er.allCells());
    const CollOfScalar source_values = (er.inputCollectionOfScalar("source_values", source_cells) * double(1));
    const CollOfScalar source = er.operatorExtend(source_values, source_cells, er.allCells());
    const CollOfScalar insource_sw = er.operatorExtend(double(1), er.allCells());
    auto sw0 = sw_initial;
    auto p0 = er.operatorExtend(double(0), er.allCells());
    er.output("pressure", p0);
    er.output("saturation", sw0);
    for (const Scalar& dt : timesteps) {
        const CollOfScalar total_mobility = (computeWaterMob_i1_(sw0) + computeOilMob_i2_(sw0));
        auto pressureResLocal = [&](const CollOfScalar& pressure) -> CollOfScalar {
            return computePressureResidual(pressure, total_mobility, source);
        };
        const CollOfScalar p = er.newtonSolve(pressureResLocal, p0);
        const CollOfScalar flux = computeTotalFlux_i8_(p, total_mobility);
        auto transportResLocal = [&](const CollOfScalar& sw) -> CollOfScalar {
            return computeTransportResidual(sw, sw0, flux, source, insource_sw, dt);
        };
        const CollOfScalar sw = er.newtonSolve(transportResLocal, er.operatorExtend(double(0.5), er.allCells()));
        p0 = p;
        sw0 = sw;
        er.output("pressure", p0);
        er.output("saturation", sw0);
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
