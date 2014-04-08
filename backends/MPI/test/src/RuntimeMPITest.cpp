#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include "equelle/RuntimeMPI.hpp"
#include "equelle/mpiutils.hpp"

using namespace equelle;

BOOST_AUTO_TEST_CASE( globalCollectionSize ) {
    equelle::RuntimeMPI runtime;

    runtime.decompose();

    BOOST_MESSAGE( "SubGrid.size: " << runtime.subGrid.global_cell.size() );
}

BOOST_AUTO_TEST_CASE( allGather ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() > 1, "Test requires program to be run with mpirun." );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    // These are global input data.
    // Rely on default grid being 6x1
    std::vector<double> a_0 = { 0, 1, 2, 3, 4, 5 };
    injectMockData( param, "a", a_0.begin(), a_0.end() );

    // Create the Equelle runtime.
    equelle::RuntimeMPI er( param );
    er.decompose();

    const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());
    const CollOfScalar a_global = er.allGather( a );

    BOOST_REQUIRE_EQUAL( er.globalGrid->c_grid()->number_of_cells, a_global.size() );

    BOOST_CHECK_EQUAL_COLLECTIONS( a_global.value().data(),  a_global.value().data() + a_global.value().size(),
                                   a_0.begin(), a_0.end() );

}

BOOST_AUTO_TEST_CASE( logging ) {    
    equelle::RuntimeMPI runtime;

    runtime.logstream << "Hello world\n";

    BOOST_CHECK_MESSAGE( true, "If this compiles, the observeable state of the program is corret");
}

