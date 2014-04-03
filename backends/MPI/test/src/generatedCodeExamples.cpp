#define BOOST_TEST_NO_MAIN

#include <fstream>

#include <boost/iterator.hpp>
#include <boost/test/unit_test.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <algorithm>
#include <numeric>

#include "equelle/RuntimeMPI.hpp"


using namespace equelle;

namespace {

// These make testing a lot easier.

std::vector<double> operator+( std::vector<double>& v0, std::vector<double>& v1 ) {
    std::vector<double> res( v0.size() );
    std::transform( v0.begin(), v0.end(), v1.begin(), res.begin(),
                    []( double a, double b ) { return a + b; } );
    return res;
}

std::vector<double> operator-( std::vector<double>& v0, std::vector<double>& v1 ) {
    std::vector<double> res( v0.size() );
    std::transform( v0.begin(), v0.end(), v1.begin(), res.begin(),
                    []( double a, double b ) { return a - b; } );
    return res;
}

std::vector<double> operator*( std::vector<double>& v0, std::vector<double>& v1 ) {
    std::vector<double> res( v0.size() );
    std::transform( v0.begin(), v0.end(), v1.begin(), res.begin(),
                    []( double a, double b ) { return a * b; } );
    return res;
}

std::vector<double> operator/( std::vector<double>& v0, std::vector<double>& v1 ) {
    std::vector<double> res( v0.size() );
    std::transform( v0.begin(), v0.end(), v1.begin(), res.begin(),
                    []( double a, double b ) { return a / b; } );
    return res;
}


}
BOOST_AUTO_TEST_CASE( basicCalculator ) {
    // Get user parameters.

    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() > 1, "Test requires program to be run with mpirun." );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    // These are global input data.
    std::vector<double> a_gold = {0, 1, 2, 3, 4, 5 };
    std::vector<double> b_gold = {5, 4, 3, 2, 1, 0 };
    std::vector<double> d_gold = {2, 2, 2, 2, 2, 2 };

    injectMockData( param, "a", a_gold.begin(), a_gold.end() );
    injectMockData( param, "b", b_gold.begin(), b_gold.end() );
    injectMockData( param, "d", d_gold.begin(), d_gold.end() );

    // Create the Equelle runtime.
    equelle::RuntimeMPI er( param );
    er.decompose();
    //ensureRequirements(er);

    // Code from examples/app/smallCals.cpp
    // ============= Generated code starts here ================
    const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());
    const CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());
    const CollOfScalar d = er.inputCollectionOfScalar("d", er.allCells());
    const CollOfScalar c = (a - b);
    const CollOfScalar e = (d - c);
    const CollOfScalar f = (e / d);
    const CollOfScalar g = (f + b);
    const CollOfScalar h = (g * d);

    // ============= Generated code ends here ================

    const auto c_global = er.allGather( c );
    const auto e_global = er.allGather( e );
    const auto f_global = er.allGather( f );
    const auto g_global = er.allGather( g );
    const auto h_global = er.allGather( h );

    auto c_gold = (a_gold - b_gold);
    auto e_gold = (d_gold - c_gold);
    auto f_gold = (e_gold / d_gold);
    auto g_gold = (f_gold + b_gold);
    auto h_gold = (g_gold * d_gold);

    BOOST_CHECK_EQUAL_COLLECTIONS( c_global.value().data(), c_global.value().data() +  6,
                                   c_gold.begin(), c_gold.end() );

    BOOST_CHECK_EQUAL_COLLECTIONS( e_global.value().data(), e_global.value().data() +  6,
                                   e_gold.begin(), e_gold.end() );

    BOOST_CHECK_EQUAL_COLLECTIONS( f_global.value().data(), f_global.value().data() +  6,
                                   f_gold.begin(), f_gold.end() );

    BOOST_CHECK_EQUAL_COLLECTIONS( g_global.value().data(), g_global.value().data() +  6,
                                   g_gold.begin(), g_gold.end() );

    BOOST_CHECK_EQUAL_COLLECTIONS( h_global.value().data(), h_global.value().data() +  6,
                                   h_gold.begin(), h_gold.end() );
}
