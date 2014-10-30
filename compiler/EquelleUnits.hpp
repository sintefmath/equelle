/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef OPM_EQUELLEUNITS_HEADER_INCLUDED
#define OPM_EQUELLEUNITS_HEADER_INCLUDED

#include "Dimension.hpp"
#include <utility>
#include <string>
#include <map>

struct UnitData
{
    UnitData()
        : valid(false)
    {
    }
    UnitData(const Dimension& dim_arg, const double conv_factor_arg)
        : dimension(dim_arg), conv_factor(conv_factor_arg), valid(true)
    {
    }
    UnitData operator*(const double factor) const
    {
        return UnitData(dimension, conv_factor * factor);
    }
    Dimension dimension;
    double conv_factor;
    bool valid;
};

inline UnitData operator*(const double factor, const UnitData& ud)
{
    return ud * factor;
}

inline std::map<std::string, UnitData> buildUnitMap()
{
    using namespace DimensionConstant;

    // Create unit map by repeated inserts.
    std::map<std::string, UnitData> u;
    // Prefixes
    u["Pico"] = UnitData(nodim, 1e-12);
    u["Nano"] = UnitData(nodim, 1e-9);
    u["Micro"] = UnitData(nodim, 1e-6);
    u["Milli"] = UnitData(nodim, 1e-3);
    u["Centi"] = UnitData(nodim, 0.01);
    u["Deci"] = UnitData(nodim, 0.1);
    u["Deca"] = UnitData(nodim, 10);
    u["Hecto"] = UnitData(nodim, 100);
    u["Kilo"] = UnitData(nodim, 1e3);
    u["Mega"] = UnitData(nodim, 1e6);
    u["Giga"] = UnitData(nodim, 1e9);
    u["Tera"] = UnitData(nodim, 1e12);
    // Length
    u["Meter"] = UnitData(length, 1.0);
    u["Inch"] = UnitData(length, 0.0254);
    u["Feet"] = 12.0 * u["Inch"];
    // Time
    u["Second"] =  UnitData(DimensionConstant::time, 1.0); // Because time is also a function in the C library.
    u["Minute"] =  60.0 * u["Second"];
    u["Hour"] =  60.0 * u["Minute"];
    u["Day"] =  24.0 * u["Hour"];
    // Mass
    u["Kilogram"] = UnitData(mass, 1.0);
    // Temperature
    u["Kelvin"] = UnitData(temperature, 1.0);

    // Derived units.
    u["Newton"] = UnitData(force, 1.0);
    u["Joule"] = UnitData(energy, 1.0);
    u["Pascal"] = UnitData(pressure, 1.0);
    u["Watt"] = UnitData(effect, 1.0);

    return u;
}

inline UnitData unitFromString(const std::string& s)
{
    static std::map<std::string, UnitData> unit_map = buildUnitMap();
    auto it = unit_map.find(s);
    if (it != unit_map.end()) {
        return it->second;
    } else {
        return UnitData();
    }
}


#endif // OPM_EQUELLEUNITS_HEADER_INCLUDED
