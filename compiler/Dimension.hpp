/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
*/

#ifndef DIMENSION_HEADER_INCLUDED
#define DIMENSION_HEADER_INCLUDED

#include <array>
#include <string>
#include <iostream>

enum BaseDimension {
    Length,
    Time,
    Mass,
    Temperature,
    ElectricCurrent,
    QuantityOfSubstance,
    LuminousIntensity
};

class Dimension
{
public:
    Dimension()
        : dim_{{ 0, 0, 0, 0, 0, 0, 0 }}
    {
    }

    // Accept implicit conversion from array.
    Dimension(const std::array<int,7>& dim)
        : dim_(dim)
    {
    }

    void setCoefficient(BaseDimension index, int coeff)
    {
        dim_[index] = coeff;
    }

    int coefficient(BaseDimension index) const
    {
        return dim_[index];
    }

    Dimension operator+ (const Dimension& other) const
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] += other.dim_[dd];
        }
        return result;
    }

    Dimension operator- (const Dimension& other) const
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] -= other.dim_[dd];
        }
        return result;
    }

    Dimension operator* (const int power) const
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] *= power;
        }
        return result;
    }

    bool operator== (const Dimension& other) const
    {
        return dim_ == other.dim_;
    }

    bool operator!= (const Dimension& other) const
    {
        return !(*this == other);
    }

private:
    typedef std::array<int,7> DA;
    DA dim_;
};


namespace DimensionConstant
{
    // Basic dimension constants.
    const Dimension nodim      {};
    const Dimension length     {{{ 1, 0, 0, 0, 0, 0, 0 }}};
    const Dimension time       {{{ 0, 1, 0, 0, 0, 0, 0 }}};
    const Dimension mass       {{{ 0, 0, 1, 0, 0, 0, 0 }}};
    const Dimension temperature{{{ 0, 0, 0, 1, 0, 0, 0 }}};

    // Derived dimension constants.
    const Dimension area = length * 2;
    const Dimension velocity = length - time;
    const Dimension acceleration = velocity - time;
    const Dimension force = mass + acceleration;
    const Dimension energy = force + length;
    const Dimension pressure = force - area;
    const Dimension effect = energy - time;
} // namespace DimensionConstant


// Needed for visit(UnitNode&).
inline std::string dimString(BaseDimension bd)
{
    switch (bd) {
    case Length:
        return "Meter";
    case Time:
        return "Second";
    case Mass:
        return "Kilogram";
    case Temperature:
        return "Kelvin";
    case ElectricCurrent:
        return "Ampere";
    case QuantityOfSubstance:
        return "Mole";
    case LuminousIntensity:
        return "Candela";
    default:
        throw std::logic_error("Error in dimString() -- unknown enum value.");
    }
}

// Needed for visit(UnitNode&).
inline std::ostream& operator<<(std::ostream& os, const Dimension& dim)
{
    if (dim == Dimension()) {
        return os;
    }
    int count = 0;
    os << "[";
    for (int i = 0; i < 7; ++i) {
        BaseDimension bd = static_cast<BaseDimension>(i);
        const int coef = dim.coefficient(bd);
        if (coef == 1) {
            if (count) {
                os << " * ";
            }
            os << dimString(bd);
            ++count;
        } else if (coef != 0) {
            if (count) {
                os << " * ";
            }
            os << dimString(bd) << "^" << coef;
            ++count;
        }
    }
    os << "]";
    return os;
}


#endif // DIMENSION_HEADER_INCLUDED
