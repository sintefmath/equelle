#pragma once

#include <iostream>
#include <fstream>

#include <opm/core/utility/ErrorMacros.hpp>

namespace equelle {


template <class SomeCollection>
CollOfScalar RuntimeMPI::inputCollectionOfScalar(const String& name,
                                                 const SomeCollection& coll)
{
    const int size = coll.size();
    const bool from_file = param_.getDefault(name + "_from_file", false);
    if (from_file) {
        const String filename = param_.get<String>(name + "_filename");
        std::ifstream is(filename.c_str());
        if (!is) {
            OPM_THROW(std::runtime_error, "Could not find file " << filename);
        }
        std::istream_iterator<double> beg(is);
        std::istream_iterator<double> end;
        std::vector<double> data(beg, end);
        if (int(data.size()) != size) {
            OPM_THROW(std::runtime_error, "Unexpected size of input data for " << name << " in file " << filename);
        }
        return CollOfScalar(CollOfScalar::V(Eigen::Map<CollOfScalar::V>(&data[0], size)));
    } else {
        // Uniform values.
        return CollOfScalar(CollOfScalar::V::Constant(size, param_.get<double>(name)));
    }
}

} // namespace equelle
