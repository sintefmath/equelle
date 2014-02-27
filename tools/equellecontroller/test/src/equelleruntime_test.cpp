#define BOSOT_TEST_MAIN
#define BOOST_TEST_MODULE EquelleControllerTest
#include <boost/test/unit_test.hpp>

#include "equelle/equellecontroller.hpp"

BOOST_AUTO_TEST_CASE( testFactoryMethod )
{
    EquelleController cont = EquelleController::createEquelleController();
}

