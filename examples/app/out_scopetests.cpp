
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

    const Scalar a = double(5);
    const Scalar b = double(6);
    auto foo = [&](const Scalar& x, const Scalar& y) -> Scalar {
        return ((a * x) + (b * y));
    };
    const SeqOfScalar seq = er.inputSequenceOfScalar("seq");
    for (const Scalar& elem : seq) {
        const Scalar r = ((a + double(3)) + elem);
        const SeqOfScalar seq2 = er.inputSequenceOfScalar("seq2");
        for (const Scalar& e2 : seq2) {
            auto foo3 = [&](const Scalar& y) -> Scalar {
                return foo(double(3), y);
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
