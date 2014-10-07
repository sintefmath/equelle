
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

    // ============= Generated code starts here ================

    const Scalar cfl = er.inputScalarWithDefault("cfl", double(0.9));
    const Scalar g = er.inputScalarWithDefault("g", double(9.81));
    const CollOfScalar h0 = er.inputCollectionOfScalar("h0", er.allCells());
    const CollOfScalar hu0 = er.inputCollectionOfScalar("hu0", er.allCells());
    const CollOfScalar hv0 = er.inputCollectionOfScalar("hv0", er.allCells());
    const std::array<CollOfScalar, 3> q0 = makeArray(h0, hu0, hv0);
    const std::array<CollOfScalar, 3> q = q0;
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> fflux = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const CollOfScalar h = q[0];
        const CollOfScalar hu = q[1];
        const CollOfScalar hv = q[2];
        return makeArray(hu, ((hu * (hu / h)) + ((((double(1) / double(2)) * g) * h) * h)), (hu * (hv / h)));
    };
    std::function<std::array<CollOfScalar, 3>(const std::array<CollOfScalar, 3>&)> gflux = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 3> {
        const CollOfScalar h = q[0];
        const CollOfScalar hu = q[1];
        const CollOfScalar hv = q[2];
        return makeArray(hu, (hu * (hv / h)), ((hv * (hv / h)) + ((((double(1) / double(2)) * g) * h) * h)));
    };
    std::function<std::array<CollOfScalar, 4>(const std::array<CollOfScalar, 3>&, const std::array<CollOfScalar, 3>&, const CollOfScalar&, const CollOfVector&)> compute_flux = [&](const std::array<CollOfScalar, 3>& ql, const std::array<CollOfScalar, 3>& qr, const CollOfScalar& l, const CollOfVector& n) -> std::array<CollOfScalar, 4> {
        const CollOfScalar hl = ql[0];
        const CollOfScalar hul = ql[1];
        const CollOfScalar hvl = ql[2];
        const CollOfScalar hr = qr[0];
        const CollOfScalar hur = qr[1];
        const CollOfScalar hvr = qr[2];
        const Scalar pl = double(0.7);
        const Scalar pr = double(0.9);
        const CollOfScalar cl = er.sqrt((g * hl));
        const CollOfScalar cr = er.sqrt((g * hr));
        const Scalar am = double(0);
        const Scalar ap = double(0);
        const std::array<Scalar, 3> f_flux = makeArray(double(0.9), double(0.9), double(0.9));
        const std::array<Scalar, 3> g_flux = makeArray(double(0.8), double(0.8), double(0.8));
        const std::array<Scalar, 3> central_upwind_correction = makeArray(double(0.9), double(0.9), double(0.9));
        const std::array<CollOfScalar, 3> flux = makeArray(er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()), er.operatorExtend(double(0.9), er.allFaces()));
        const CollOfScalar max_wave_speed = er.operatorExtend(double(0.8), er.allFaces());
        return makeArray(flux[0], flux[1], flux[2], max_wave_speed);
    };
    std::function<std::array<CollOfScalar, 2>(const std::array<CollOfScalar, 3>&)> reconstruct_plane = [&](const std::array<CollOfScalar, 3>& q) -> std::array<CollOfScalar, 2> {
        return makeArray(er.operatorExtend(double(0), er.allCells()), er.operatorExtend(double(0), er.allCells()));
    };
    const CollOfFace ifs = er.interiorFaces();
    const CollOfCell first = er.firstCell(ifs);
    const CollOfCell second = er.secondCell(ifs);
    const std::array<CollOfScalar, 2> slopes = reconstruct_plane(q);
    const CollOfVector n = er.normal(ifs);
    const CollOfVector ip = er.centroid(ifs);
    const CollOfVector first_to_ip = (ip - er.centroid(first));
    const CollOfVector second_to_ip = (ip - er.centroid(second));
    const CollOfScalar l = er.norm(ifs);
    const std::array<CollOfScalar, 3> q1 = makeArray(er.operatorOn(q[0], er.allCells(), first), er.operatorOn(q[1], er.allCells(), first), er.operatorOn(q[2], er.allCells(), first));
    const std::array<CollOfScalar, 3> q2 = makeArray(er.operatorOn(q[0], er.allCells(), second), er.operatorOn(q[1], er.allCells(), second), er.operatorOn(q[2], er.allCells(), second));
    const std::array<CollOfScalar, 4> flux_and_max_wave_speed = compute_flux(q1, q2, l, n);
    const Scalar min_area = double(0.9);
    const Scalar max_wave_speed = double(0.8);
    const Scalar dt = (cfl * (min_area / (double(6) * max_wave_speed)));
    er.output("q0", q[0]);
    er.output("q1", q[1]);
    er.output("q2", q[2]);

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
