
#include <string>
#include <iostream>
#include <vector>

#include <opm/core/utility/ErrorMacros.hpp>

#include "cos_soon.hpp"
#include "CudaArray.hpp"


using namespace equelleCUDA;

CollOfScalarSoon::CollOfScalarSoon() 
    : val_()
{
    // Intentionally left empty
}

CollOfScalarSoon::CollOfScalarSoon(const int size)
    : val_(size)
{
    // Intentionally left empty
}

CollOfScalarSoon::CollOfScalarSoon(const int size, const double value)
    : val_(size, value)
{
    // Intentionally left emtpy
}

CollOfScalarSoon::CollOfScalarSoon(const CudaArray& val)
    : val_(val)
{
    // Intentionally left emtpy
}

CollOfScalarSoon::CollOfScalarSoon(const std::vector<double>& host_vec)
    : val_(host_vec)
{
    // Intentionally left emtpy
}

// Copy constructor
CollOfScalarSoon::CollOfScalarSoon(const CollOfScalarSoon& coll)
    : val_(coll.val_)
{
    // Intentionally left emtpy
}

// Assignment copy operator
CollOfScalarSoon& CollOfScalarSoon::operator= (const CollOfScalarSoon& other)
{
    // Protect against self assignment:
    if (this != &other) {
	val_ = other.val_;
    }
    return *this;
}
   
CollOfScalarSoon::~CollOfScalarSoon()
{
    // Intentionally left blank as val_ knows how to delete itself.
}



// Member functions that only have to return val_'s function:
const double* CollOfScalarSoon::data() const {
    return val_.data();
}

double* CollOfScalarSoon::data() {
    return val_.data();
}

kernelSetup CollOfScalarSoon::setup() const {
    return val_.setup();
}

std::vector<double> CollOfScalarSoon::copyToHost() const {
    return val_.copyToHost();
}

int CollOfScalarSoon::size() const {
    return val_.size();
}




// Get referance to the CudaArray with the values:
const CudaArray& CollOfScalarSoon::val() const {
    return val_;
}

CollOfScalarSoon equelleCUDA::operator+ (const CollOfScalarSoon& lhs,
					 const CollOfScalarSoon& rhs)
{
    CudaArray val = lhs.val_ + rhs.val_;
    std::cout << "pluss completed\n";
    return CollOfScalarSoon(val);
}