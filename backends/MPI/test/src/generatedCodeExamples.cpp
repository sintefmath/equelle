#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

#include "equelle/RuntimeMPI.hpp"


BOOST_AUTO_TEST_CASE( distributedReadFile ) {
    // Get user parameters.
    //Opm::parameter::ParameterGroup param;//(argc, argv, false);

    // Create the Equelle runtime.
    equelle::RuntimeMPI er;//( param );

    //ensureRequirements(er);

    // ============= Generated code starts here ================

    //const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());

/*    const CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());
    const CollOfScalar d = er.inputCollectionOfScalar("d", er.allCells());
    const CollOfScalar c = (a - b);
    const CollOfScalar e = (d - c);
    er.output("e", e);
    const CollOfScalar f = (e / d);
    er.output("f", f);
    const CollOfScalar g = (f + b);
    er.output("g", g);
    const CollOfScalar h = (g * d);
    er.output("h", h);
*/
    // ============= Generated code ends here ================
}
