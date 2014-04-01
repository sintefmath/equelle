#define BOOST_TEST_NO_MAIN

#include <fstream>

#include <boost/iterator.hpp>
#include <boost/test/unit_test.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <algorithm>
#include <numeric>

#include "equelle/RuntimeMPI.hpp"

namespace {
    template<class T>
    void injectMockData( Opm::parameter::ParameterGroup& param, std::string key, T begin, T end ) {
        std::string filename = key + ".mockdata";
        param.insertParameter( key + "_from_file", "true" );
        param.insertParameter( key + "_filename", filename );

        std::ofstream f(filename);
        std::copy( begin, end, std::ostream_iterator<typename T::value_type>( f, " " ) );
    }

}

BOOST_AUTO_TEST_CASE( basicCalculator ) {
    // Get user parameters.

    BOOST_REQUIRE_MESSAGE( equelle::getMPISize() > 1, "Test requires program to be run with mpirun." );
    Opm::parameter::ParameterGroup param;
    param.disableOutput();

    // These are global input data.
    std::vector<double> a_0 = { 0, 1, 2, 3, 4, 5 };
    std::vector<double> b_0 = {5, 4, 3, 2, 1, 0 };
    std::vector<double> d_0 = {2, 2, 2, 2, 2, 2 };

    injectMockData( param, "a", a_0.begin(), a_0.end() );
    injectMockData( param, "b", b_0.begin(), b_0.end() );
    injectMockData( param, "d", d_0.begin(), d_0.end() );

    // Create the Equelle runtime.
    equelle::RuntimeMPI er( param );
    er.decompose();
    //ensureRequirements(er);

    using namespace equelle;

    // ============= Generated code starts here ================
    BOOST_MESSAGE( er.subGrid.global_cell.size() );
    const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());
    const CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());

    const CollOfScalar c = (a - b);
    std::vector<double> c_global = {-5, -3, -1, 1, 3, 5 };

    for( auto i = 0; i < er.subGrid.global_cell.size(); ++i ) {
        auto gid = er.subGrid.global_cell[i];
        BOOST_CHECK_EQUAL( c.value()[i], c_global[gid] );
    }

    const CollOfScalar d = er.inputCollectionOfScalar("d", er.allCells());

    const CollOfScalar e = (d - c);

    std::vector<double> e_global = { 2+5, 2+3, 2+1, 2-1, 2-3, 2-5 };
    for( auto i = 0; i < er.subGrid.global_cell.size(); ++i ) {
        auto gid = er.subGrid.global_cell[i];
        BOOST_CHECK_EQUAL( e.value()[i], e_global[gid] );
    }

    const CollOfScalar f = (e / d);

    std::vector<double> f_global;
    std::transform( e_global.begin(), e_global.end(), d_0.begin(), std::back_inserter( f_global ),
                    []( double e, double d ) { return e / d; } );

    for( auto i = 0; i < er.subGrid.global_cell.size(); ++i ) {
        auto gid = er.subGrid.global_cell[i];
        BOOST_CHECK_EQUAL( f.value()[i], f_global[gid] );
    }

/*    const CollOfScalar d = er.inputCollectionOfScalar("d", er.allCells());
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
