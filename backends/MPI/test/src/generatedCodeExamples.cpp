#define BOOST_TEST_NO_MAIN

#include <fstream>

#include <boost/test/unit_test.hpp>
#include <opm/core/utility/parameters/ParameterGroup.hpp>

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

BOOST_AUTO_TEST_CASE( distributedReadFile ) {
    // Get user parameters.
    Opm::parameter::ParameterGroup param;

    std::vector<double> v = {0, 1, 2, 3, 4, 5 };

    injectMockData( param, "a", v.begin(), v.end() );

    // Create the Equelle runtime.
    equelle::RuntimeMPI er( param );
    er.decompose();
    //ensureRequirements(er);

    using namespace equelle;

    // ============= Generated code starts here ================
    BOOST_MESSAGE( er.subGrid.global_cell.size() );
    //const CollOfScalar a = er.inputCollectionOfScalar("a", er.allCells());

    //const CollOfScalar b = er.inputCollectionOfScalar("b", er.allCells());
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
