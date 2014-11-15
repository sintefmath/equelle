
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

    const Scalar rsp = double(287.058);
    const Scalar temp = double(290);
    const Scalar perm = double(9.8692e-13);
    const Scalar mobility = double(52500);
    const CollOfScalar q = (er.inputCollectionOfScalar("source", er.allCells()) * double(1));
    const SeqOfScalar timesteps = (er.inputSequenceOfScalar("timesteps") * double(1));
    const CollOfScalar p_initial = er.operatorExtend(double(3000000), er.allCells());
    const CollOfFace intf = er.interiorFaces();
    const CollOfCell f = er.firstCell(intf);
    const CollOfCell s = er.secondCell(intf);
    const CollOfScalar area = er.norm(intf);
    const CollOfScalar vol = er.norm(er.allCells());
    const CollOfVector d1 = (er.centroid(f) - er.centroid(intf));
    const CollOfVector d2 = (er.centroid(s) - er.centroid(intf));
    const CollOfScalar h1 = ((-area * perm) * (er.dot(er.normal(intf), d1) / er.dot(d1, d1)));
    const CollOfScalar h2 = ((area * perm) * (er.dot(er.normal(intf), d2) / er.dot(d2, d2)));
    const CollOfScalar trans = (double(1) / ((double(1) / h1) + (double(1) / h2)));
    auto density_i0_ = [&](const CollOfScalar& p) -> CollOfScalar {
        return (p / (rsp * temp));
    };
    auto density_i1_ = [&](const CollOfScalar& p) -> CollOfScalar {
        return (p / (rsp * temp));
    };
    auto residual = [&](const CollOfScalar& p, const CollOfScalar& p0, const Scalar& dt) -> CollOfScalar {
        const CollOfScalar v = ((mobility * trans) * (er.operatorOn(p, er.allCells(), f) - er.operatorOn(p, er.allCells(), s)));
        const CollOfScalar rho = density_i0_(p);
        const CollOfScalar rho0 = density_i1_(p0);
        const CollOfScalar rho_face = ((er.operatorOn(rho, er.allCells(), f) + er.operatorOn(rho, er.allCells(), s)) / double(2));
        const CollOfScalar res = ((((vol / dt) * (rho - rho0)) + er.divergence((v * rho_face))) - q);
        return res;
    };
    auto p0 = p_initial;
    for (const Scalar& dt : timesteps) {
        auto locRes = [&](const CollOfScalar& p) -> CollOfScalar {
            return residual(p, p0, dt);
        };
        const CollOfScalar p = er.newtonSolve(locRes, p0);
        er.output("pressure", p);
        p0 = p;
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
