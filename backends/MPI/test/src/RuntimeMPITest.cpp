#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include "equelle/RuntimeMPI.hpp"
#include "equelle/EquelleRuntimeCPU.hpp"
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


BOOST_AUTO_TEST_CASE( inputScalarWithDefault ) {
    Opm::parameter::ParameterGroup param;

    param.insertParameter( "a", "42.0" );

    equelle::RuntimeMPI er( param );
    er.decompose();

    const Scalar k = er.inputScalarWithDefault("k", double(0.3));
    const Scalar a = er.inputScalarWithDefault("a", double(-1.0));

    BOOST_CHECK_CLOSE( k, 0.3, 1e-7 );
    BOOST_CHECK_CLOSE( a, 42.0, 1e-7 );
}


BOOST_AUTO_TEST_CASE( boundaryCells ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() > 1, "Test requires program to be run with mpirun." );
    Opm::parameter::ParameterGroup param;

    // Ensure we have at least one interior cell.
    param.insertParameter( "nx", "3" );
    param.insertParameter( "ny", "3" );

    equelle::RuntimeMPI er( param );
    er.decompose();

    equelle::EquelleRuntimeCPU ser( param );
    CollOfCell gold_global_boundary = ser.boundaryCells();

    // Boundary return indices in node-enumeration so we must map them into the global index space.
    CollOfCell boundary = er.boundaryCells();

    BOOST_CHECK( !boundary.empty() );

    auto global_boundary = er.subGrid.map_to_global( boundary );

    er.logstream << "all cells (local coordinate system)";
    for( auto c: er.allCells() ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;

    er.logstream << "local cells: ";
    for( auto c: boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;

    er.logstream << "global cells: ";
    for( auto c: global_boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;


    // Assert all the boundary cells are in the global boundary
    for( auto c: global_boundary ) {
        auto it = std::find( gold_global_boundary.begin(), gold_global_boundary.end(), c );
        BOOST_CHECK_MESSAGE( it != global_boundary.end(),
                     "global cell " << c.index << " is not in the global boundary");
    }

    er.logstream << "gold global boundary: ";
    for( auto c: gold_global_boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;
}


BOOST_AUTO_TEST_CASE( logging ) {    
    equelle::RuntimeMPI runtime;

    runtime.logstream << "Hello world\n";

    BOOST_CHECK_MESSAGE( true, "If this compiles, the observeable state of the program is corret");
}

