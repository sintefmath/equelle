/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED
#define EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED


#include <fstream>
#include <iterator>
#include <opm/core/utility/StopWatch.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>


template <class EntityCollection>
CollOfScalarCPU EquelleRuntimeCUDA::operatorExtend(const double data,
                                               const EntityCollection& to_set)
{
    return CollOfScalarCPU();
    //return CollOfScalarCPU(CollOfScalarCPU::V::Constant(to_set.size(), data));
}


namespace
{
    using Opm::subset;
    using Opm::superset;

    template <class EntityCollection>
    std::vector<int> subsetIndices(const EntityCollection& superset,
                                   const EntityCollection& subset)
    {
        if (subset.empty()) {
            return std::vector<int>();
        }

        assert(std::is_sorted(superset.begin(), superset.end()));
        assert(std::adjacent_find(superset.begin(), superset.end()) == superset.end());
        assert(superset[0].index >= 0);

        const std::size_t sub_sz = subset.size();
        typedef typename EntityCollection::value_type Entity;
        std::vector<std::pair<Entity, int> > sub_indexed(sub_sz);
        for (std::size_t elem = 0; elem < sub_sz; ++elem) {
            sub_indexed[elem] = std::make_pair(subset[elem], elem);
        }
        std::sort(sub_indexed.begin(), sub_indexed.end());
        assert(sub_indexed[0].first.index >= 0);

        const std::size_t super_sz = superset.size();
        std::vector<int> indices(sub_sz);
        std::size_t sub_index = 0;
        for (std::size_t i = 0; i < super_sz; ++i) {
            while (sub_indexed[sub_index].first == superset[i]) {
                indices[sub_indexed[sub_index].second] = i;
                ++sub_index;
            }
            if (sub_index == sub_sz) {
                // All subset elements have been matched.
                break;
            }
        }
#if 0
        // Debugging output.
        std::cout << "Superset:\n";
        for (auto e : superset) {
            std::cout << e.index << ' ';
        }
        std::cout << std::endl;
        std::cout << "Subset:\n";
        for (auto e : subset) {
            std::cout << e.index << ' ';
        }
        std::cout << std::endl;
        std::cout << "Indices:\n";
        for (auto i : indices) {
            std::cout << i << ' ';
        }
        std::cout << std::endl;
        std::cout << "Sizes = " << indices.size() << ' ' << subset.size() << std::endl;
#endif
        assert(sub_index == sub_sz);
        return indices;
    }

    template <class EntityType, class IntVec>
    std::vector<EntityType> subset(const std::vector<EntityType>& x,
                                   const IntVec& indices)
    {
        const size_t sz = indices.size();
        std::vector<EntityType> retval;
        retval.reserve(sz);
        for (size_t i = 0; i < sz; ++i) {
            retval.push_back(x[indices[i]]);
        }
        return retval;
    }

    template <class EntityType, class IntVec>
    std::vector<EntityType> superset(const std::vector<EntityType>& x,
                                     const IntVec& indices,
                                     const int n)
    {
        assert(x.size() == indices.size());
        const size_t sz = indices.size();
        std::vector<EntityType> retval(n);
        for (size_t i = 0; i < sz; ++i) {
            retval[indices[i]] = x[i];
        }
        return retval;
    }
} // anon namespace

template <class SomeCollection, class EntityCollection>
SomeCollection EquelleRuntimeCUDA::operatorExtend(const SomeCollection& data,
                                                 const EntityCollection& from_set,
                                                 const EntityCollection& to_set)
{
    assert(size_t(data.size()) == size_t(from_set.size()));
    // Expand with zeros.
    std::vector<int> indices = subsetIndices(to_set, from_set);
    assert(indices.size() == from_set.size());
    return superset(data, indices, to_set.size());
}





/*template <class SomeCollection>
CollOfScalarCPU EquelleRuntimeCUDA::inputCollectionOfScalar(const String& name,
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
        return CollOfScalarCPU(CollOfScalarCPU::V(Eigen::Map<CollOfScalarCPU::V>(&data[0], size)));
    } else {
        // Uniform values.
        return CollOfScalarCPU(CollOfScalarCPU::V::Constant(size, param_.get<double>(name)));
    }
}*/


#endif // EQUELLERUNTIMECUDA_IMPL_HEADER_INCLUDED
