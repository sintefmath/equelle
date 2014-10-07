/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
*/

#ifndef DIMENSION_HEADER_INCLUDED
#define DIMENSION_HEADER_INCLUDED

#include <array>

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

    int coefficient(BaseDimension index)
    {
        return dim_[index];
    }

    Dimension operator+ (const Dimension& other)
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] += other.dim_[dd];
        }
        return result;
    }

    Dimension operator- (const Dimension& other)
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] -= other.dim_[dd];
        }
        return result;
    }

    Dimension operator* (const int power)
    {
        Dimension result(*this);
        for (int dd = 0; dd < 7; ++dd) {
            result.dim_[dd] *= power;
        }
        return result;
    }

    bool operator== (const Dimension& other)
    {
        return dim_ == other.dim_;
    }

    bool operator!= (const Dimension& other)
    {
        return !(*this == other);
    }

private:
    std::array<int,7> dim_;
};

#endif // DIMENSION_HEADER_INCLUDED
