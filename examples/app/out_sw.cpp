
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

    const Scalar cfl = er.inputScalarWithDefault("cfl", double(0.9));
    const Scalar g = er.inputScalarWithDefault("g", double(9.81));
    const CollOfScalar h0 = er.inputCollectionOfScalar("h0", er.allCells());
    const CollOfScalar hu0 = er.inputCollectionOfScalar("hu0", er.allCells());
    const CollOfScalar hv0 = er.inputCollectionOfScalar("hv0", er.allCells());
    const auto q0 = makeArray(h0, hu0, hv0);
    const auto q = q0;
    auto fflux = [&](const auto& q) {
        const CollOfScalar h = std::get<0>(q);
        const CollOfScalar hu = std::get<1>(q);
        const CollOfScalar hv = std::get<2>(q);
        return makeArray(hu, ((hu * (hu / h)) + ((((double(1) / double(2)) * g) * h) * h)), (hu * (hv / h)));
    };
    auto gflux = [&](const auto& q) {
        const CollOfScalar h = std::get<0>(q);
        const CollOfScalar hu = std::get<1>(q);
        const CollOfScalar hv = std::get<2>(q);
        return makeArray(hu, (hu * (hv / h)), ((hv * (hv / h)) + ((((double(1) / double(2)) * g) * h) * h)));
    };
    auto compute_flux = [&](const auto& ql, const auto& qr, const auto& l, const auto& n) {
        const CollOfScalar hl = std::get<0>(ql);
        const CollOfScalar hul = std::get<1>(ql);
        const CollOfScalar hvl = std::get<2>(ql);
        const CollOfScalar hr = std::get<0>(qr);
        const CollOfScalar hur = std::get<1>(qr);
        const CollOfScalar hvr = std::get<2>(qr);
        const Scalar pl = double(0.7);
        const Scalar pr = double(0.9);
        const CollOfScalar cl = er.sqrt((g * hl));
        const CollOfScalar cr = er.sqrt((g * hr));
        const Scalar am = double(0);
        const Scalar ap = double(0);
        const auto f_flux = makeArray(double(0.9), double(0.9), double(0.9));
        const auto g_flux = makeArray(double(0.8), double(0.8), double(0.8));
        const auto central_upwind_correction = makeArray(double(0.9), double(0.9), double(0.9));
        const auto flux = makeArray(er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()));
        const CollOfScalar max_wave_speed = er.operatorExtend(double(0.8), er.allFaces());
        return makeArray(std::get<0>(flux), std::get<1>(flux), std::get<2>(flux), max_wave_speed);
    };
    auto reconstruct_plane = [&](const auto& q) {
        return makeArray(er.operatorExtend(double(0), er.allCells()), er.operatorExtend(double(0), er.allCells()));
    };
    const CollOfFace ifs = er.interiorFaces();
    const CollOfCell first = er.firstCell(ifs);
    const CollOfCell second = er.secondCell(ifs);
    const auto slopes = reconstruct_plane(q);
    const CollOfVector n = er.normal(ifs);
    const CollOfVector ip = er.centroid(ifs);
    const CollOfVector first_to_ip = (ip - er.centroid(first));
    const CollOfVector second_to_ip = (ip - er.centroid(second));
    const CollOfScalar l = er.norm(ifs);
    const auto q1 = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), first), er.operatorOn(std::get<1>(q), er.allCells(), first), er.operatorOn(std::get<2>(q), er.allCells(), first));
    const auto q2 = makeArray(er.operatorOn(std::get<0>(q), er.allCells(), second), er.operatorOn(std::get<1>(q), er.allCells(), second), er.operatorOn(std::get<2>(q), er.allCells(), second));
    const auto flux_and_max_wave_speed = compute_flux(q1, q2, l, n);
    const Scalar min_area = double(0.9);
    const Scalar max_wave_speed = double(0.8);
    const Scalar dt = (cfl * (min_area / (double(6) * max_wave_speed)));
    er.output("q0", std::get<0>(q));
    er.output("q1", std::get<1>(q));
    er.output("q2", std::get<2>(q));

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
