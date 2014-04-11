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


// In order to look at the outputs to the logstream (an easy way to do MPI printf debuggin)
// it is recomenned to run the test application using the --run_test=boundaryCells
// command line option.
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

    er.logstream << "grid is " << param.getDefault("nx", 0) << "x" << param.getDefault("ny", 0) << std::endl;
    er.logstream << "all cells (local coordinate system)";
    for( auto c: er.allCells() ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;


    er.logstream << "all cells (global coordinate system)";
    for( auto c: er.subGrid.map_to_global( er.allCells() ) ) {
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
        BOOST_CHECK_MESSAGE( it != gold_global_boundary.end(),
                     "global cell " << c.index << " is not in the global boundary");
    }

    er.logstream << "gold global boundary: ";
    for( auto c: gold_global_boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;
}

BOOST_AUTO_TEST_CASE( boundaryFaces ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() > 1, "Test requires program to be run with mpirun." );
    Opm::parameter::ParameterGroup param;

    // Ensure we have at least one interior cell.
    param.insertParameter( "nx", "6" );
    param.insertParameter( "ny", "1" );

    equelle::RuntimeMPI er( param );
    er.decompose();

    equelle::EquelleRuntimeCPU ser( param );
    CollOfFace gold_global_boundary = ser.boundaryFaces();

    auto boundary = er.boundaryFaces();

    // All cells are on the boundary in the 6x1 grid so it should not be empty.
    BOOST_CHECK( !boundary.empty() );

    auto global_boundary = er.subGrid.map_to_global( boundary );

    er.logstream << "all faces (local coordinate system)";
    for( auto c: er.allFaces() ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;

    er.logstream << "local faces: ";
    for( auto c: boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;

    er.logstream << "global face: ";
    for( auto c: global_boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;


    // Assert all the boundary cells are in the global boundary
    for( auto c: global_boundary ) {
        auto it = std::find( gold_global_boundary.begin(), gold_global_boundary.end(), c );
        BOOST_CHECK_MESSAGE( it != gold_global_boundary.end(),
                     "global face " << c.index << " is not in the global boundary");
    }

    er.logstream << "gold global boundary: ";
    for( auto c: gold_global_boundary ) {
        er.logstream << c.index << ", ";
    }
    er.logstream << std::endl;
}

BOOST_AUTO_TEST_CASE( inputDomainSubsetOf_faces ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() == 2, "Test requires program to be run on exactly two nodes" );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    param.insertParameter( "nx", "6" );
    param.insertParameter( "ny", "1" );

    // These faces are on the boundary, and face 9 is on both nodes (as part of the southern face of global cell 2.
    std::vector<int> global_dirichlet_boundary = { 7, 8, 9 };
    injectMockData( param, "dirichlet_boundary", global_dirichlet_boundary.begin(), global_dirichlet_boundary.end() );

    equelle::RuntimeMPI er( param );
    er.decompose();

    CollOfFace dirichlet_boundary = er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces() );

    // This could be dependent on the partitioning from Zoltan.
    if ( equelle::getMPIRank() == 0 ) {
        BOOST_CHECK_EQUAL( dirichlet_boundary.size(),    3 );
    } else if ( equelle::getMPIRank() == 1 ) {
        BOOST_CHECK_EQUAL( dirichlet_boundary.size(), 1 );
    }
}

BOOST_AUTO_TEST_CASE( inputDomainSubsetOf_cells ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() == 2, "Test requires program to be run on exactly two nodes" );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    // Ensure we have at least one interior cell - which is cell 4.
    param.insertParameter( "nx", "3" );
    param.insertParameter( "ny", "3" );

    std::vector<int> global_subRegion = { 0 }; // This should only go to node 0, even with ghost cells.
                                                  // (But this could be dependent on the partitioning....)
    injectMockData( param, "subRegion", global_subRegion.begin(), global_subRegion.end() );

    equelle::RuntimeMPI er( param );
    er.decompose();

    CollOfCell subRegion = er.inputDomainSubsetOf("subRegion", er.boundaryCells() );

    // Check that the global id of the subRegion is 0
    for( auto x: subRegion ) {
        auto localId = x.index;
        BOOST_CHECK_EQUAL( er.subGrid.global_cell[ localId ], 0 );
    }
}

BOOST_AUTO_TEST_CASE( inputDomainSubsetOf_invalid_superset ) {
    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() == 2, "Test requires program to be run on exactly two nodes" );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();


    param.insertParameter( "nx", "6" );
    param.insertParameter( "ny", "1" );

    // These faces are on the boundary, and face 9 is on both nodes (as part of the southern face of global cell 2.
    // In addition we add face 2, which is an interor node but exists on both partitions
    std::vector<int> global_dirichlet_boundary = { 2, 7, 8, 9 };
    injectMockData( param, "dirichlet_boundary", global_dirichlet_boundary.begin(), global_dirichlet_boundary.end() );

    equelle::RuntimeMPI er( param );
    er.decompose();

    BOOST_CHECK_THROW( er.inputDomainSubsetOf("dirichlet_boundary", er.boundaryFaces() ),
                       std::runtime_error );
}


BOOST_AUTO_TEST_CASE( logging ) {    
    equelle::RuntimeMPI runtime;

    runtime.logstream << "Hello world\n";

    BOOST_CHECK_MESSAGE( true, "If this compiles, the observeable state of the program is corret");
}

