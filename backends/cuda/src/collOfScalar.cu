
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
    CudaArray val = lhs.val_ + rhs.val_;
    std::cout << "pluss completed\n";
    return CollOfScalar(val);
}



CollOfScalar equelleCUDA::operator-(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    kernelSetup s = out.setup();
    minus_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

//CollOfScalar equelleCUDA::operator+(const CollOfScalar& lhs, const CollOfScalar& rhs) {
//
//    CollOfScalar out = lhs;
//    kernelSetup s = out.setup();
//    plus_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
//    return out;
//}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    kernelSetup s = out.setup();
    multiplication_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const CollOfScalar& rhs) {

    CollOfScalar out = lhs;
    kernelSetup s = out.setup();
    division_kernel <<<s.grid, s.block>>>(out.data(), rhs.data(), out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const Scalar lhs, const CollOfScalar& rhs) {
    CollOfScalar out = rhs;
    kernelSetup s = out.setup();
    multScalCollection_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator*(const CollOfScalar& lhs, const Scalar rhs) {
    return (rhs * lhs);
}

CollOfScalar equelleCUDA::operator/(const CollOfScalar& lhs, const Scalar rhs) {
    return ( (1/rhs) * lhs);
}

CollOfScalar equelleCUDA::operator/(const Scalar lhs, const CollOfScalar& rhs) {
    CollOfScalar out = rhs;
    kernelSetup s = out.setup();
    divScalCollection_kernel<<<s.grid,s.block>>>(out.data(), lhs, out.size());
    return out;
}

CollOfScalar equelleCUDA::operator-(const CollOfScalar& arg) {
    return -1.0*arg;
}


//  >
CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGTscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>(const Scalar lhs, const CollOfScalar& rhs) {
    CollOfBool out(rhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = rhs.setup();
    comp_scalGTcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs, rhs.data(), rhs.size());
    return out;
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
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGEcoll_kernel<<<s.grid,s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>=(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collGEscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator>=(const Scalar lhs, const CollOfScalar& rhs) {
    CollOfBool out(rhs.size());
    bool* out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = rhs.setup();
    comp_scalGEcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs, rhs.data(), rhs.size());
    return out;
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
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collEQcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator==(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collEQscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator==(const Scalar lhs, const CollOfScalar& rhs) {
    return (rhs == lhs);
}


// !=
CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const CollOfScalar& rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collNEcoll_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs.data(), lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator!=(const CollOfScalar& lhs, const Scalar rhs) {
    CollOfBool out(lhs.size());
    bool *out_ptr = thrust::raw_pointer_cast( &out[0] );
    kernelSetup s = lhs.setup();
    comp_collNEscal_kernel<<<s.grid, s.block>>>(out_ptr, lhs.data(), rhs, lhs.size());
    return out;
}

CollOfBool equelleCUDA::operator!=(const Scalar lhs, const CollOfScalar& rhs) {
    return (rhs != lhs);
}