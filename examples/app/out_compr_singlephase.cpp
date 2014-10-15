
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

    const auto rsp = double(287.058);
    const auto temp = double(290);
    const auto perm = double(9.8692e-13);
    const auto mobility = double(52500);
    const auto q = er.inputCollectionOfScalar("source", er.allCells());
    const auto timesteps = er.inputSequenceOfScalar("timesteps");
    const auto p_initial = er.operatorExtend(double(3000000), er.allCells());
    const auto intf = er.interiorFaces();
    const auto f = er.firstCell(intf);
    const auto s = er.secondCell(intf);
    const auto area = er.norm(intf);
    const auto vol = er.norm(er.allCells());
    const auto d1 = (er.centroid(f) - er.centroid(intf));
    const auto d2 = (er.centroid(s) - er.centroid(intf));
    const auto h1 = ((-area * perm) * (er.dot(er.normal(intf), d1) / er.dot(d1, d1)));
    const auto h2 = ((area * perm) * (er.dot(er.normal(intf), d2) / er.dot(d2, d2)));
    const auto trans = (double(1) / ((double(1) / h1) + (double(1) / h2)));
    auto density = [&](const auto& p) {
        return (p / (rsp * temp));
    };
    auto residual = [&](const auto& p, const auto& p0, const auto& dt) {
        const auto v = ((mobility * trans) * (er.operatorOn(p, er.allCells(), f) - er.operatorOn(p, er.allCells(), s)));
        const auto rho = density(p);
        const auto rho0 = density(p0);
        const auto rho_face = ((er.operatorOn(rho, er.allCells(), f) + er.operatorOn(rho, er.allCells(), s)) / double(2));
        const auto res = ((((vol / dt) * (rho - rho0)) + er.divergence((v * rho_face))) - q);
        return res;
    };
    auto p0 = p_initial;
    for (const Scalar& dt : timesteps) {
        auto locRes = [&](const auto& p) {
            return residual(p, p0, dt);
        };
        const auto p = er.newtonSolve(locRes, p0);
        er.output("pressure", p);
        p0 = p;
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
