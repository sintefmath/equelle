#define BOOST_TEST_MODULE CollOfScalarClass

#include <boost/test/included/unit_test.hpp>

#include <CollOfScalar.hpp>
#include <EquelleRuntimeCUDA_havahol.hpp>
#include <EquelleRuntimeCUDA.hpp>

#include <vector>

using namespace equelleCUDA;

static void compareVectors( std::vector<double> answer, std::vector<double> lf)
{
    BOOST_REQUIRE_EQUAL_COLLECTIONS( answer.begin(), answer.end(),
				     lf.begin(), lf.end() );
}

BOOST_AUTO_TEST_CASE ( testingTheTester )
{
    std::vector<double> a;
    a.push_back(2);
    a.push_back(4);
    
    std::vector<double> b;
    b.push_back(2);
    b.push_back(4);

    compareVectors( a, b );    
}

BOOST_AUTO_TEST_SUITE( arithmetics );

BOOST_AUTO_TEST_CASE( pluss_tests )
{
    int size = 10000;
    std::vector<double> a, b, lf;
    for (int i = 0; i < size; ++i) {
	a.push_back(i);
	b.push_back(i*2 + 10);
	lf.push_back( a[i] + b[i] );
    }
    CollOfScalar CoS_a(a);
    CollOfScalar CoS_b(b);
    CollOfScalar res = CoS_a + CoS_b;
    compareVectors( res.copyToHost(), lf);
}


BOOST_AUTO_TEST_CASE( minus_test )
{
    //int size = 10000;
    int size = 1024*1024;
    std::vector<double> a, b, lf;
    for(int i = 0; i < size; ++i) {
	a.push_back( i*(i%4 + 1));
	b.push_back( i - 20 );
	lf.push_back( a[i] - b[i]);
    }
    CollOfScalar cos_a(a);
    CollOfScalar cos_b(b);
    CollOfScalar res = cos_a - cos_b ;
    compareVectors( res.copyToHost(), lf);
}


BOOST_AUTO_TEST_CASE( multiplication_test )
{
    int size = 10000;
    std::vector<double> a, b, lf;
    for(int i = 0; i < size; ++i) {
	a.push_back( i*(i%4 + 1));
	b.push_back( i - 20 );
	lf.push_back( a[i] * b[i]);
    }
    CollOfScalar cos_a(a);
    CollOfScalar cos_b(b);
    CollOfScalar res = cos_a * cos_b ;
    compareVectors( res.copyToHost(), lf);
}

BOOST_AUTO_TEST_CASE( division_test )
{
    int size = 10000;
    std::vector<double> a, b, lf;
    for(int i = 0; i < size; ++i) {
	a.push_back( i*(i%4 + 1));
	b.push_back( i + 20 );
	lf.push_back( a[i] / b[i]);
    }
    CollOfScalar cos_a(a);
    CollOfScalar cos_b(b);
    CollOfScalar res = cos_a / cos_b ;
    compareVectors( res.copyToHost(), lf);
}

BOOST_AUTO_TEST_SUITE_END();
