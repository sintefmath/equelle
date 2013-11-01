
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

void ensureRequirements(const EquelleRuntimeCPU& er);

int main(int argc, char** argv)
{
    // Get user parameters.
    Opm::parameter::ParameterGroup param(argc, argv, false);

    // Create the Equelle runtime.
    EquelleRuntimeCPU er(param);

    ensureRequirements(er);

    // ============= Generated code starts here ================

    const Scalar a = double(5);
    const Scalar b = double(6);
    auto foo = [&](const Scalar& x, const Scalar& y) -> Scalar {
        return ((a * x) + (b * y));
    };
    const SeqOfScalar seq = er.userSpecifiedSequenceOfScalar("seq");
    for (const Scalar& elem : seq) {
        const Scalar r = ((a + double(3)) + elem);
        const SeqOfScalar seq2 = er.userSpecifiedSequenceOfScalar("seq2");
        for (const Scalar& e2 : seq2) {
            auto foo3 = [&](const Scalar& y) -> Scalar {
                auto three = [&]() -> Scalar {
                    return double(3);
                };
                return foo(three(), y);
            };
            const Scalar q = ((b + foo3(e2)) + r);
            er.output("q", q);
        }
    }

    // ============= Generated code ends here ================

    return 0;
}

void ensureRequirements(const EquelleRuntimeCPU& er)
{
}
