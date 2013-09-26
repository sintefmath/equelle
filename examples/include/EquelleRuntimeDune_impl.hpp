/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef EQUELLERUNTIMEDUNE_IMPL_HEADER_INCLUDED
#define EQUELLERUNTIMEDUNE_IMPL_HEADER_INCLUDED


template <class EntityCollection>
CollOfScalars EquelleRuntimeDune::operatorOn(const Scalar data,
                                             const EntityCollection& to_set)
{
    return CollOfScalars::Constant(to_set.size(), data);
}


namespace
{
    template <class IntVec>
    CollOfScalars
    subset(const CollOfScalars& x,
           const IntVec& indices)
    {
        CollOfScalars retval = CollOfScalars::Zero(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            retval[i] = x[indices[i]];
        }
        return retval;
    }

    template <class IntVec>
    CollOfScalars
    superset(const CollOfScalars& x,
             const IntVec& indices,
             const int sz)
    {
        CollOfScalars retval = CollOfScalars::Zero(sz);
        for (size_t i = 0; i < indices.size(); ++i) {
            retval[indices[i]] = x[i];
        }
        return retval;
    }

    template <class EntityCollection>
    std::vector<int> subsetIndices(const EntityCollection& superset,
                                   const EntityCollection& subset)
    {
        const size_t super_sz = superset.size();
        const size_t sub_sz = subset.size();
        assert(sub_sz <= super_sz);
        std::vector<int> indices;
        indices.reserve(sub_sz);
        size_t sub_index = 0;
        for (size_t i = 0; i < super_sz; ++i) {
            if (subset[sub_index] == superset[i]) {
                indices.push_back(i);
                ++sub_index;
            }
        }
        assert(indices.size() == sub_sz);
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
SomeCollection EquelleRuntimeDune::operatorOn(const SomeCollection& data,
                                              const EntityCollection& from_set,
                                              const EntityCollection& to_set)
{
    // The implementation assumes that from_set is a subset of to_set, or vice versa.
    const size_t from_sz = data.size();
    assert(from_sz == from_set.size());
    const size_t to_sz = to_set.size();
    if (to_sz == from_sz) {
        assert(from_set == to_set);
        return data;
    } else if (to_sz > from_sz) {
        // Expand with zeros.
        std::vector<int> indices = subsetIndices(to_set, from_set);
        assert(indices.size() == from_sz);
        return superset(data, indices, to_sz);
    } else {
        // Extract subset.
        std::vector<int> indices = subsetIndices(from_set, to_set);
        assert(indices.size() == to_sz);
        return subset(data, indices);
    }
}



template <class SomeCollection>
SomeCollection EquelleRuntimeDune::trinaryIf(const CollOfBooleans& predicate,
                                             const SomeCollection& iftrue,
                                             const SomeCollection& iffalse) const
{
    const size_t sz = predicate.size();
    assert(sz == iftrue.size() && sz == iffalse.size());
    SomeCollection retval = iftrue;
    for (size_t i = 0; i < sz; ++i) {
        if (!predicate[i]) {
            retval[i] = iffalse[i];
        }
    }
    return retval;
}



#endif // EQUELLERUNTIMEDUNE_IMPL_HEADER_INCLUDED
