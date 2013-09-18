/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.

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

#ifndef OPM_EQUELLERUNTIMECPU_IMPL_HEADER_INCLUDED
#define OPM_EQUELLERUNTIMECPU_IMPL_HEADER_INCLUDED


template <class EntityCollection>
CollOfScalars EquelleRuntimeCPU::operatorOn(const double data,
                                            const EntityCollection& to_set)
{
    return CollOfScalars::Constant(to_set.size(), data);
}


namespace
{
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
SomeCollection EquelleRuntimeCPU::operatorOn(const SomeCollection& data,
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
SomeCollection EquelleRuntimeCPU::trinaryIf(const CollOfBooleans& predicate,
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


template <>
inline CollOfScalarsAD EquelleRuntimeCPU::trinaryIf(const CollOfBooleans& predicate,
                                             const CollOfScalarsAD& iftrue,
                                             const CollOfScalarsAD& iffalse) const
{
    const size_t sz = predicate.size();
    assert(sz == iftrue.size() && sz == iffalse.size());
    CollOfScalars trueones = CollOfScalars::Constant(sz, 1.0);
    CollOfScalars falseones = CollOfScalars::Constant(sz, 0.0);
    for (size_t i = 0; i < sz; ++i) {
        if (!predicate[i]) {
            trueones[i] = 0.0;
            falseones[i] = 1.0;
        }
    }
    CollOfScalarsAD retval = iftrue * trueones + iffalse * falseones;
    return retval;
}


template <class ResidualFunctor>
CollOfScalarsAD EquelleRuntimeCPU::newtonSolve(const ResidualFunctor& rescomp,
					       const CollOfScalars& u_initialguess) const
{
    // Set up Newton loop.
    CollOfScalarsAD u = singlePrimaryVariable(u_initialguess);
    output("Initial u", u);
    output("norm", twoNorm(u));
    CollOfScalarsAD residual = rescomp(u); //  Generated code in here
    output("Initial residual", residual);
    output("norm", twoNorm(residual));
    const int max_iter = 10;
    const double tol = 1e-6;
    int iter = 0;

    // Execute newton loop until residual is small or we have used too many iterations.
    while ( (twoNorm(residual) > tol) && (iter < max_iter) ) {
        // Debugging output not specified in Equelle.
        std::cout << "\niter = " << iter << " (max = " << max_iter
                  << "), norm(residual) = " << twoNorm(residual)
                  << " (tol = " << tol << ")" << std::endl;

        // Solve linear equations for du, apply update.
        const CollOfScalars du = solveForUpdate(residual);
        u = u - du;

        // Recompute residual.
        residual = rescomp(u);

        // Debugging output not specified in Equelle.
        output("u", u);
        output("norm(u)", twoNorm(u));
        output("residual", residual);
        output("norm(residual)", twoNorm(residual));

        ++iter;
    }
    return u;
}

#endif // OPM_EQUELLERUNTIMECPU_IMPL_HEADER_INCLUDED
