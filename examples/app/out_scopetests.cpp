
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

    const Scalar a = double(5);
    const Scalar b = double(6);
    std::function<Scalar(const Scalar&, const Scalar&)> foo = [&](const Scalar& x, const Scalar& y) -> Scalar {
        return ((a * x) + (b * y));
    };
    const SeqOfScalar seq = er.inputSequenceOfScalar("seq");
    for (const Scalar& elem : seq) {
        const Scalar r = ((a + double(3)) + elem);
        const SeqOfScalar seq2 = er.inputSequenceOfScalar("seq2");
        for (const Scalar& e2 : seq2) {
            std::function<Scalar(const Scalar&)> foo3 = [&](const Scalar& y) -> Scalar {
                std::function<Scalar()> three = [&]() -> Scalar {
                    return double(3);
                };
                return foo(three(), y);
            };
            const Scalar q = ((b + foo3(e2)) + r);
            er.output("q", q);
        }
    }

    // ============= Generated code ends here ================

}

void ensureRequirements(const equelle::EquelleRuntimeCPU& er)
{
    (void)er;
}
