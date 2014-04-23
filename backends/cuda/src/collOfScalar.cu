
#include <string>
#include <iostream>
#include <vector>

#include <opm/core/utility/ErrorMacros.hpp>

#include "CollOfScalar.hpp"
#include "CudaArray.hpp"
#include "CudaMatrix.hpp"


using namespace equelleCUDA;
using namespace wrapCudaArray;

CollOfScalar::CollOfScalar() 
    : val_(),
      der_(),
      autodiff_(false)
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size)
    : val_(size),
      der_(),
      autodiff_(false)
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size, const double value)
    : val_(size, value),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const CudaArray& val)
    : val_(val),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : val_(host_vec),
      der_(),
      autodiff_(false)
{
    // Intentionally left emtpy
}


// Primary variable constructor
CollOfScalar::CollOfScalar(const CollOfScalar& val, const bool primaryVariable)
    : val_(val.val_),
      der_(val.size()),
      autodiff_(true)
{
    // It makes no sence to use this constructor with primaryVariable = false,
    // so we check that it is used correctly:
    if ( !primaryVariable ) {
	OPM_THROW(std::runtime_error, "Trying to create a primary variable with primaryVarible = " << primaryVariable );
    }
}

// Constructor from CudaArray and CudaMatrix
CollOfScalar::CollOfScalar(const CudaArray& val, const CudaMatrix& der)
    : val_(val),
      der_(der),
      autodiff_(true)
{
    // Intentionally left empty
}


// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll)
    : val_(coll.val_),
      der_(coll.der_),
      autodiff_(coll.autodiff_)
{
    // Intentionally left emtpy
}

// Assignment copy operator
CollOfScalar& CollOfScalar::operator= (const CollOfScalar& other)
{
    // Protect against self assignment:
    if (this != &other) {
	val_ = other.val_;
	autodiff_ = other.autodiff_;
	if ( autodiff_ ) {
	    der_ = other.der_;
	}
    }
    return *this;
}
   
CollOfScalar::~CollOfScalar()
{
    // Intentionally left blank as val_ knows how to delete itself.
}



// Member functions that only have to return val_'s function:
const double* CollOfScalar::data() const {
    return val_.data();
}

double* CollOfScalar::data() {
    return val_.data();
}

bool CollOfScalar::useAutoDiff() const {
    return autodiff_;
}

kernelSetup CollOfScalar::setup() const {
    return val_.setup();
}

std::vector<double> CollOfScalar::copyToHost() const {
    return val_.copyToHost();
}

hostMat CollOfScalar::matrixToHost() const {
    if ( !autodiff_ ) {
	OPM_THROW(std::runtime_error, "Trying to copy empty matrix to host\n");
    }
    return der_.toHost();
}

int CollOfScalar::size() const {
    return val_.size();
}

CudaMatrix CollOfScalar::derivative() const {
    return der_;
}

CudaArray CollOfScalar::value() const {
    return val_;
}



// Get referance to the CudaArray with the values:
//const CudaArray& CollOfScalar::val() const {
//    return val_;
//}




// ------- ARITHMETIC OPERATIONS --------------------

CollOfScalar equelleCUDA::operator+ (const CollOfScalar& lhs,
				     const CollOfScalar& rhs)
{
    //CudaArray val = lhs.val_ + rhs.val_;
    std::cout << "pluss completed\n";
    CollOfScalar out;
    out.val_ = lhs.val_ + rhs.val_;
    if (lhs.autodiff_ || rhs.autodiff_) {
	out.autodiff_ = true;
	out.der_ = lhs.der_ + rhs.der_;
    }
    return out;
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out;
    out.val_ = lhs.val_ - rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
	out.autodiff_ = true;
	out.der_ = lhs.der_ - rhs.der_;
    }
    return out;
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfScalar out;
    out.val_ = lhs.val_ * rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
	out.autodiff_ = true;
	// (u*v)' = u'*v + v'*u = diag(v)*u' + diag(u)*v'
	// where u = lhs and v = rhs
	CudaMatrix diag_u(lhs.val_);
	CudaMatrix diag_v(rhs.val_);
	std::cout << "Finding derivative of product\n";
	out.der_ = diag_v*lhs.der_ + diag_u*rhs.der_;
    }
    return out;
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    // Need AD
    CollOfScalar out;
    out.val_ = lhs.val_ / rhs.val_;
    if ( lhs.autodiff_ || rhs.autodiff_ ) {
	out.autodiff_ = true;
	// (u/v)' = (u'*v - v'*u)/(v^2)
	// where u = lhs and v = rhs
	CudaMatrix diag_u(lhs.val_); // D1 
	CudaMatrix diag_v(rhs.val_); // D2
	CudaMatrix inv_v_squared( 1.0/(rhs.val_ * rhs.val_));
	out.der_ = inv_v_squared*( diag_v*lhs.der_ - diag_u*rhs.der_);
    }
    return out;
}

CollOfScalar equelleCUDA::operator*(const Scalar lhs, const CollOfScalar& rhs) {
    CollOfScalar out;
    out.val_ = lhs * rhs.val_;
    if ( rhs.autodiff_ ) {
	out.autodiff_ = true;
	out.der_ = lhs * rhs.der_;
    }
    return out; 
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const Scalar rhs) {
    return ( rhs * lhs);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const Scalar rhs) {
    return ( (1/rhs) * lhs);
}

CollOfScalar equelleCUDA::operator/(const Scalar lhs, const CollOfScalar& rhs) {
    // Need AD
    CollOfScalar out;
    out.val_ = lhs / rhs.val_;
    if ( rhs.autodiff_ ) {
	out.autodiff_ = true;
	// (a/u)' = - (a/u^2)*u'
	// where a = lhs and u = rhs
	CudaMatrix diag_u_squared(lhs/(rhs.val_ * rhs.val_));
	out.der_ = - diag_u_squared*rhs.der_;
    }
    return out;
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg) {
    return -1.0*arg;
}


//  >
CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return ( lhs.val_ > rhs.val_ );
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const Scalar rhs) {
    return ( lhs.val_ > rhs );
}

CollOfBool equelleCUDA::operator>(const Scalar lhs, const CollOfScalar& rhs) {
    return ( lhs > rhs.val_ );
}


// <
CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    // if   a < b   then b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const CollOfScalar& lhs, const Scalar rhs) {
    // if  a < b  then   b > a
    return rhs > lhs;
}

CollOfBool equelleCUDA::operator<(const Scalar lhs, const CollOfScalar& rhs) {
    // if   a < b   then b > a
    return rhs > lhs;
}


// >=
CollOfBool equelleCUDA::operator>=(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return ( lhs.val_ >= rhs.val_ );
}

CollOfBool equelleCUDA::operator>=(const CollOfScalar& lhs, const Scalar rhs) {
    return ( lhs.val_ >= rhs );
}

CollOfBool equelleCUDA::operator>=(const Scalar lhs, const CollOfScalar& rhs) {
    return ( lhs >= rhs.val_ );
}


// <= 
CollOfBool equelleCUDA::operator<=(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    // if   a <= b   then b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const CollOfScalar& lhs, const Scalar rhs) {
    // if  a <= b  then   b >= a
    return rhs >= lhs;
}

CollOfBool equelleCUDA::operator<=(const Scalar lhs, const CollOfScalar& rhs) {
    // if   a <= b   then b >= a
    return rhs >= lhs;
}


// ==
CollOfBool equelleCUDA::operator==(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return ( lhs.val_ == rhs.val_ );
}

CollOfBool equelleCUDA::operator==(const CollOfScalar& lhs, const Scalar rhs) {
    return ( lhs.val_ == rhs );
}

CollOfBool equelleCUDA::operator==(const Scalar lhs, const CollOfScalar& rhs) {
    return (rhs == lhs);
}


// !=
CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return ( lhs.val_ != rhs.val_ );
}

CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const Scalar rhs) {
    return ( lhs.val_ != rhs );
}

CollOfBool equelleCUDA::operator!=(const Scalar lhs, const CollOfScalar& rhs) {
    return (rhs != lhs);
}