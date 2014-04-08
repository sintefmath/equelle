
#include <string>
#include <iostream>
#include <vector>

#include <opm/core/utility/ErrorMacros.hpp>

#include "CollOfScalar.hpp"
#include "CudaArray.hpp"


using namespace equelleCUDA;
using namespace wrapCudaArray;

CollOfScalar::CollOfScalar() 
    : val_()
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size)
    : val_(size)
{
    // Intentionally left empty
}

CollOfScalar::CollOfScalar(const int size, const double value)
    : val_(size, value)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const CudaArray& val)
    : val_(val)
{
    // Intentionally left emtpy
}

CollOfScalar::CollOfScalar(const std::vector<double>& host_vec)
    : val_(host_vec)
{
    // Intentionally left emtpy
}

// Copy constructor
CollOfScalar::CollOfScalar(const CollOfScalar& coll)
    : val_(coll.val_)
{
    // Intentionally left emtpy
}

// Assignment copy operator
CollOfScalar& CollOfScalar::operator= (const CollOfScalar& other)
{
    // Protect against self assignment:
    if (this != &other) {
	val_ = other.val_;
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

kernelSetup CollOfScalar::setup() const {
    return val_.setup();
}

std::vector<double> CollOfScalar::copyToHost() const {
    return val_.copyToHost();
}

int CollOfScalar::size() const {
    return val_.size();
}




// Get referance to the CudaArray with the values:
const CudaArray& CollOfScalar::val() const {
    return val_;
}




// ------- ARITHMETIC OPERATIONS --------------------

CollOfScalar equelleCUDA::operator+ (const CollOfScalar& lhs,
					 const CollOfScalar& rhs)
{
    //CudaArray val = lhs.val_ + rhs.val_;
    std::cout << "pluss completed\n";
    return CollOfScalar(lhs.val_ + rhs.val_);
}



CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return CollOfScalar( lhs.val_ - rhs.val_);
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return CollOfScalar(lhs.val_ * rhs.val_);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    return CollOfScalar(lhs.val_ / rhs.val_);
}

CollOfScalar equelleCUDA::operator*(const Scalar lhs, const CollOfScalar& rhs) {
    return CollOfScalar( lhs * rhs.val_);
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const Scalar rhs) {
    return ( rhs * lhs);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const Scalar rhs) {
    return ( (1/rhs) * lhs);
}

CollOfScalar equelleCUDA::operator/(const Scalar lhs, const CollOfScalar& rhs) {
    return CollOfScalar( lhs / rhs.val_ );
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