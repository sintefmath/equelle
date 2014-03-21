#define BOOST_TEST_NO_MAIN

#include <boost/test/unit_test.hpp>
#include "equelle/RuntimeMPI.hpp"


BOOST_AUTO_TEST_CASE( globalCollectionSize ) {
    equelle::RuntimeMPI runtime;

    runtime.decompose();

    BOOST_MESSAGE( "SubGrid.size: " << runtime.subGrid.global_cell.size() );



}
