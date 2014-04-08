
// This file implements test for the following:
// - CollOfScalar without arithmetics without use of AutoDiff
// - CollOfScalar comparisons.




#define BOOST_TEST_MODULE CollOfScalarClass

#include <boost/test/included/unit_test.hpp>

#include <stdlib.h>

#include "CollOfScalar.hpp"
#include "equelleTypedefs.hpp"
#include "EquelleRuntimeCUDA_havahol.hpp"
#include "EquelleRuntimeCUDA.hpp"

#include <vector>

using namespace equelleCUDA;

const int ALL_SIZES = 10000;

static void compareVectors( std::vector<double> answer, std::vector<double> lf)
{
    BOOST_REQUIRE_EQUAL_COLLECTIONS( answer.begin(), answer.end(),
				     lf.begin(), lf.end() );
}

static void compareBools( std::vector<bool> answer, std::vector<bool> lf)
{
    BOOST_REQUIRE_EQUAL_COLLECTIONS( answer.begin(), answer.end(),
				     lf.begin(), lf.end());
}

static void compareVectorsDiv( std::vector<double> answer, std::vector<double> lf)
{
    for (int i = 0; i < lf.size(); i++) {
	BOOST_REQUIRE_CLOSE( answer[i], lf[i], 0.000000000001 );
    }
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
    //int size = ALL_SIZES;
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
    int size = 10000;
    //int size = 1024*1024;
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

BOOST_AUTO_TEST_CASE( scal_coll_multiplication_test )
{
    int size = 10000;
    std::vector<double> a, lf;
    double myDoub = 1.15;
    for (int i = 0; i < size; ++i) {
	a.push_back( i*2.25);
	lf.push_back( i*2.25*myDoub );
    }
    CollOfScalar col_a(a);
    CollOfScalar res = myDoub * col_a;
    compareVectors ( res.copyToHost(), lf);
}

BOOST_AUTO_TEST_CASE( coll_scal_multiplication_test )
{
    int size = 10000;
    std::vector<double> a, lf;
    double myDoub = 5;
    for (int i = 0; i < size; ++i) {
	a.push_back( i*2.25);
	lf.push_back( i*2.25*myDoub );
    }
    CollOfScalar col_a(a);
    CollOfScalar res = col_a * myDoub;
    compareVectors ( res.copyToHost(), lf);
}

BOOST_AUTO_TEST_CASE( coll_scal_division_test )
{
    int size = 10000;
    std::vector<double> a, lf;
    double myDoub = 4.25;
    for (int i = 0; i < size; ++i) {
	a.push_back( i*2.5);
	//lf.push_back( i*2.4999999999/myDoub );
	lf.push_back( i*2.5/myDoub);
    }
    CollOfScalar col_a(a);
    CollOfScalar res = col_a / myDoub;
    compareVectorsDiv ( res.copyToHost(), lf);
}

BOOST_AUTO_TEST_CASE( scal_coll_division_test )
{
    int size = 10000;
    std::vector<double> a, lf;
    double myDoub = 4.25;
    for (int i = 0; i < size; ++i) {
	a.push_back( i%100 + 1);
	//lf.push_back( i*2.4999999999/myDoub );
	lf.push_back( myDoub/a[i]);
    }
    CollOfScalar col_a(a);
    CollOfScalar res = myDoub / col_a;
    compareVectorsDiv ( res.copyToHost(), lf);
}



BOOST_AUTO_TEST_CASE( unary_minus_test )
{
    int size = 10000;
    std::vector<double> a, lf;
    for (int i = 0; i < size; ++i) {
	a.push_back( (i%619)*2.12124);
	lf.push_back( -a[i] );
    }
    CollOfScalar col_a(a);
    CollOfScalar res = - col_a;
    compareVectors ( res.copyToHost(), lf);
}


// > tests
BOOST_AUTO_TEST_CASE( greater_than_test )
{
    int size = ALL_SIZES;
    std::vector<double> a, b;
    std::vector<bool> lf;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	b.push_back( rand() % 87 );
	lf.push_back( a[i] > b[i]);
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = a_col > b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);

}

BOOST_AUTO_TEST_CASE( greater_than_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a;
    std::vector<bool> lf;
    double b = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	lf.push_back( a[i] > b);
    }
    CollOfScalar a_col(a);
    CollOfBool cob = a_col > b;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( scalar_greater_than_test )
{
    int size = ALL_SIZES;
    std::vector<double> b;
    std::vector<bool> lf;
    double a = (rand()%113)*1.124;
    for (int i = 0; i < size; ++i) {
	b.push_back( rand() % 125);
	lf.push_back( a > b[i] );
    }
    CollOfScalar b_col(b);
    CollOfBool cob = a > b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res );
}


// < tests
BOOST_AUTO_TEST_CASE( less_than_test )
{
    int size = ALL_SIZES;
    std::vector<double> a, b;
    std::vector<bool> lf;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	b.push_back( rand() % 87 );
	lf.push_back( a[i] < b[i]);
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = a_col < b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( less_than_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a;
    std::vector<bool> lf;
    double b = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	lf.push_back( a[i] < b);
    }
    CollOfScalar a_col(a);
    CollOfBool cob = a_col < b;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( scalar_less_than_test )
{
    int size = ALL_SIZES;
    std::vector<double> b;
    std::vector<bool> lf;
    double a = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	b.push_back( rand() % 124 );
	lf.push_back( a < b[i]);
    }
    CollOfScalar b_col(b);
    CollOfBool cob = a < b_col ;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}


// >= tests:
BOOST_AUTO_TEST_CASE( greater_than_equal_test )
{
    int size = ALL_SIZES;
    std::vector<double> a, b;
    std::vector<bool> lf;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	b.push_back( rand() % 87 );
	lf.push_back( a[i] >= b[i]);
	//if ( (b[i] < a[i]) != lf[i] ) {
	//    std::cout << "Error: a[i] = " << a[i] << ", b[i] = " << b[i];
	//    std::cout << " -  a[i] >= b[i] = " << (a[i] >= b[i]) << " but ";
	//    std::cout << " b[i] < a[i] = " << (b[i] < a[i]) <<  std::endl;
	//}
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = a_col >= b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);

}

BOOST_AUTO_TEST_CASE( greater_than_equal_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a;
    std::vector<bool> lf;
    double b = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	lf.push_back( a[i] >= b);
    }
    CollOfScalar a_col(a);
    CollOfBool cob = a_col >= b;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( scalar_greater_than_equal_test )
{
    int size = ALL_SIZES;
    std::vector<double> b;
    std::vector<bool> lf;
    double a = (rand()%113)*1.124;
    for (int i = 0; i < size; ++i) {
	b.push_back( rand() % 125);
	lf.push_back( a >= b[i] );
    }
    CollOfScalar b_col(b);
    CollOfBool cob = a >= b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res );
}




// <= tests
BOOST_AUTO_TEST_CASE( less_than_equal_test )
{
    int size = ALL_SIZES;
    std::vector<double> a, b;
    std::vector<bool> lf;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	b.push_back( rand() % 87 );
	lf.push_back( a[i] <= b[i]);
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = a_col <= b_col;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( less_than_equal_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a;
    std::vector<bool> lf;
    double b = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	a.push_back( rand() % 124 );
	lf.push_back( a[i] <= b);
    }
    CollOfScalar a_col(a);
    CollOfBool cob = a_col <= b;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( scalar_less_than_equal_test )
{
    int size = ALL_SIZES;
    std::vector<double> b;
    std::vector<bool> lf;
    double a = (rand()%113)*1.244;
    for (int i = 0; i < size; ++i) {
	b.push_back( rand() % 124 );
	lf.push_back( a <= b[i]);
    }
    CollOfScalar b_col(b);
    CollOfBool cob = a <= b_col ;
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

// == tests
BOOST_AUTO_TEST_CASE( equal_test )
{
    int size = ALL_SIZES;
    std::vector<double> a(size, 0);
    std::vector<double> b(size, 0);
    std::vector<bool> lf(size, false);
    int equals = 0;
    while (equals == 0) {
	for (int i = 0; i < size; ++i ) {
	    a[i] = rand() % 37;
	    b[i] = rand() % 37;
	    lf[i] = ( a[i] == b[i] );
	    if ( lf[i] ) {
		equals++;
	    }
	}
	std::cout << "Coll == Coll: " << equals << std::endl;
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = (a_col == b_col);
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( equal_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a(size, 0);
    double b = rand() % 37;
    std::vector<bool> lf(size, false);
    int equals = 0;
    while (equals == 0) {
	for (int i = 0; i < size; ++i ) {
	    a[i] = rand() % 37;
	    lf[i] = ( a[i] == b );
	    if ( lf[i] ) {
		equals++;
	    }
	}
	std::cout << "Coll == Scal: " << equals << std::endl;
    }
    CollOfScalar a_col(a);
    CollOfBool cob = (a_col == b);
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
    
    // Add the opposite test here as well:
    CollOfBool cob_opposite = (b == a_col);
    std::vector<bool> res_opposite = cob_to_std(cob_opposite);
    compareBools( lf, res_opposite);
}


// != tests
BOOST_AUTO_TEST_CASE( inequal_test )
{
    int size = ALL_SIZES;
    std::vector<double> a(size, 0);
    std::vector<double> b(size, 0);
    std::vector<bool> lf(size, false);
    int equals = 0;
    while (equals == 0) {
	for (int i = 0; i < size; ++i ) {
	    a[i] = rand() % 37;
	    b[i] = rand() % 37;
	    lf[i] = ( a[i] != b[i] );
	    if ( !lf[i] ) {
		equals++;
	    }
	}
	std::cout << "Coll != Coll: " << equals << std::endl;
    }
    CollOfScalar a_col(a);
    CollOfScalar b_col(b);
    CollOfBool cob = (a_col != b_col);
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
}

BOOST_AUTO_TEST_CASE( inequal_scalar_test )
{
    int size = ALL_SIZES;
    std::vector<double> a(size, 0);
    double b = rand() % 37;
    std::vector<bool> lf(size, false);
    int equals = 0;
    while (equals == 0) {
	for (int i = 0; i < size; ++i ) {
	    a[i] = rand() % 37;
	    lf[i] = ( a[i] != b );
	    if ( !lf[i] ) {
		equals++;
	    }
	}
	std::cout << "Coll != Scal: " << equals << std::endl;
    }
    CollOfScalar a_col(a);
    CollOfBool cob = (a_col != b);
    std::vector<bool> res = cob_to_std(cob);
    compareBools( lf, res);
    
    // Add the opposite test here as well:
    CollOfBool cob_opposite = (b != a_col);
    std::vector<bool> res_opposite = cob_to_std(cob_opposite);
    compareBools( lf, res_opposite);
}


BOOST_AUTO_TEST_SUITE_END();

