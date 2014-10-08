/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#pragma once

#include <fstream>
#include <iterator>
#include <opm/core/utility/StopWatch.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>

namespace equelle {

template <class EntityCollection>
CollOfScalar EquelleRuntimeCPU::operatorExtend(const double data,
                                               const EntityCollection& to_set)
{
    return CollOfScalar(CollOfScalar::V::Constant(to_set.size(), data));
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
SomeCollection EquelleRuntimeCPU::operatorExtend(const SomeCollection& data,
                                                 const EntityCollection& from_set,
                                                 const EntityCollection& to_set)
{
    assert(size_t(data.size()) == size_t(from_set.size()));
    // Expand with zeros.
    std::vector<int> indices = subsetIndices(to_set, from_set);
    assert(indices.size() == from_set.size());
    return superset(data, indices, to_set.size());
}



template <class SomeCollection, class EntityCollection>
typename CollType<SomeCollection>::Type
EquelleRuntimeCPU::operatorOn(const SomeCollection& data,
                              const EntityCollection& from_set,
                              const EntityCollection& to_set)
{
    // The implementation assumes that to_set is a subset of from_set,
    // in the sense that all (possibly repeated) elements of to_set
    // are found in from_set.
    assert(size_t(data.size()) == size_t(from_set.size()));
    // Extract subset.
    std::vector<int> indices = subsetIndices(from_set, to_set);
    assert(indices.size() == to_set.size());
    return subset(data, indices);
}



template <class SomeCollection1, class SomeCollection2>
typename CollType<SomeCollection1>::Type
EquelleRuntimeCPU::trinaryIf(const CollOfBool& predicate,
                             const SomeCollection1& iftrue,
                             const SomeCollection2& iffalse) const
{
    const size_t sz = predicate.size();
    assert(sz == size_t(iftrue.size()) && sz == size_t(iffalse.size()));
    SomeCollection1 retval = iftrue;
    for (size_t i = 0; i < sz; ++i) {
        if (!predicate[i]) {
            retval[i] = iffalse[i];
        }
    }
    return retval;
}


template <>
inline CollOfScalar EquelleRuntimeCPU::trinaryIf<CollOfScalar, CollOfScalar>(const CollOfBool& predicate,
                                                                             const CollOfScalar& iftrue,
                                                                             const CollOfScalar& iffalse) const
{
    const int sz = predicate.size();
    assert(sz == iftrue.size() && sz == iffalse.size());
    CollOfScalar::V trueones = CollOfScalar::V::Constant(sz, 1.0);
    CollOfScalar::V falseones = CollOfScalar::V::Constant(sz, 0.0);
    for (int i = 0; i < sz; ++i) {
        if (!predicate[i]) {
            trueones[i] = 0.0;
            falseones[i] = 1.0;
        }
    }
    CollOfScalar retval = iftrue * trueones + iffalse * falseones;
    return retval;
}

template <>
inline CollOfScalar
EquelleRuntimeCPU::trinaryIf<CollOfScalar::ADB, CollOfScalar>(const CollOfBool& predicate,
                                                              const CollOfScalar::ADB& iftrue,
                                                              const CollOfScalar& iffalse) const
{
    return trinaryIf<CollOfScalar, CollOfScalar>(predicate, iftrue, iffalse);
}

template <>
inline CollOfScalar
EquelleRuntimeCPU::trinaryIf<CollOfScalar, CollOfScalar::ADB>(const CollOfBool& predicate,
                                                              const CollOfScalar& iftrue,
                                                              const CollOfScalar::ADB& iffalse) const
{
    return trinaryIf<CollOfScalar, CollOfScalar>(predicate, iftrue, iffalse);
}

template <>
inline CollOfScalar
EquelleRuntimeCPU::trinaryIf<CollOfScalar::ADB, CollOfScalar::ADB>(const CollOfBool& predicate,
                                                                   const CollOfScalar::ADB& iftrue,
                                                                   const CollOfScalar::ADB& iffalse) const
{
    return trinaryIf<CollOfScalar, CollOfScalar>(predicate, iftrue, iffalse);
}


template <class ResidualFunctor>
CollOfScalar EquelleRuntimeCPU::newtonSolve(const ResidualFunctor& rescomp,
                                            const CollOfScalar& u_initialguess)
{
    Opm::time::StopWatch clock;
    clock.start();

    // Set up Newton loop.
    CollOfScalar u = singlePrimaryVariable(u_initialguess);
    if (verbose_ > 2) {
        output("Initial u", u);
        output("    newtonSolve: norm (initial u)", twoNorm(u));
    }
    CollOfScalar residual = rescomp(u);
    if (verbose_ > 2) {
        output("Initial residual", residual);
        output("    newtonSolve: norm (initial residual)", twoNorm(residual));
    }

    int iter = 0;

    // Debugging output not specified in Equelle.
    if (verbose_ > 1) {
        std::cout << "    newtonSolve: iter = " << iter << " (max = " << max_iter_
                  << "), norm(residual) = " << twoNorm(residual)
                  << " (tol = " << abs_res_tol_ << ")" << std::endl;
    }

    // Execute newton loop until residual is small or we have used too many iterations.
    while ( (twoNorm(residual) > abs_res_tol_) && (iter < max_iter_) ) {

        // Solve linear equations for du, apply update.
        const CollOfScalar du = solveForUpdate(residual);
        u = u - du;

        // Recompute residual.
        residual = rescomp(u);

        if (verbose_ > 2) {
            // Debugging output not specified in Equelle.
            output("u", u);
            output("    newtonSolve: norm(u)", twoNorm(u));
            output("residual", residual);
            output("    newtonSolve: norm(residual)", twoNorm(residual));
        }

        ++iter;

        // Debugging output not specified in Equelle.
        if (verbose_ > 1) {
            std::cout << "    newtonSolve: iter = " << iter << " (max = " << max_iter_
                      << "), norm(residual) = " << twoNorm(residual)
                      << " (tol = " << abs_res_tol_ << ")" << std::endl;
        }

    }
    if (verbose_ > 0) {
        if (twoNorm(residual) > abs_res_tol_) {
            std::cout << "Newton solver failed to converge in " << max_iter_ << " iterations" << std::endl;
        } else {
            std::cout << "Newton solver converged in " << iter << " iterations" << std::endl;
        }
    }

    if (verbose_ > 1) {
        std::cout << "Newton solver took: " << clock.secsSinceLast() << " seconds." << std::endl;
    }

    return u.value();
}


template <class ... ResFuncs, class ... Colls>
std::tuple<Colls...> EquelleRuntimeCPU::newtonSolveSystem(const std::tuple<ResFuncs...>& rescomp_arg,
                                                          const std::tuple<Colls...>& u_initialguess_arg)
{
    static_assert(sizeof...(ResFuncs) == sizeof...(Colls), "Size of residual function and initial guess arrays must be identical.");
    enum { Num = sizeof ... (ResFuncs) };

    typedef std::function<CollOfScalar(const CollOfScalar&, const CollOfScalar&)> SingleResFunc;
    std::array<SingleResFunc, 2> rescomp{{std::get<0>(rescomp_arg), std::get<1>(rescomp_arg)}};
    std::array<CollOfScalar, 2> u_initialguess{{std::get<0>(u_initialguess_arg), std::get<1>(u_initialguess_arg)}};

    // Set up ranges object.
    std::array<ESpan, Num> ranges{{ ESpan(0), ESpan(0) }}; // Dummy spans that will be overwritten.
    int start = 0;
    for (int i = 0; i < Num; ++i) {
        const int end = start + u_initialguess[i].size();
        ranges[i] = ESpan(end - start, 1, start);
        start = end;
    }
    const int total_size = start;
    std::array<CollOfScalar, Num> temp;
    std::array<CollOfScalar, Num> tempres;

    // Build combined functor.
    auto combined_rescomp = [&](const CollOfScalar& u) -> CollOfScalar {
        // Split into components.
        for (int i = 0; i < Num; ++i) {
            temp[i] = subset(u, ranges[i]);
        }
        // Call each part.
        for (int i = 0; i < Num; ++i) {
            static_assert(Num == 2, "Only systems of 2 equations can be solved."); // Todo: figure out how to do Num arguments in op() below.
            tempres[i] = rescomp[i](temp[0], temp[1]);
        }
        // Recombine
        CollOfScalar result = superset(tempres[0], ranges[0], total_size);
        for (int i = 1; i < Num; ++i) {
            result += superset(tempres[i], ranges[i], total_size);
        }
        return result;
    };

    // Build combined initial guess.
    CollOfScalar combined_u_initialguess = superset(u_initialguess[0], ranges[0], total_size);
    for (int i = 1; i < Num; ++i) {
        combined_u_initialguess += superset(u_initialguess[i], ranges[i], total_size);
    }
    std::cout << "Done with setup of combined things." << std::endl;

    // Call regular Newton solver with combined objects.
    CollOfScalar combined_u = newtonSolve(combined_rescomp, combined_u_initialguess);

    // Extract subparts and return.
    for (int i = 0; i < Num; ++i) {
        temp[i] = subset(combined_u, ranges[i]);
    }
    return temp;
}


template <class SomeCollection>
CollOfScalar EquelleRuntimeCPU::inputCollectionOfScalar(const String& name,
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

