/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

// Obtain grid.
// Create a 1d cartesian grid.

// Read user parameters.
// dt, k

// Set up Newton loop.
// - convergence criteria
// - assemble linear system
//    - set up AD variables (since we know the primary unknown)
//    - arithmetic computation of residual (and Jacobian from AD)
//         -----> This is the important part!
// - solve linear system
// - update solution

// Output solution
// u

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/linalg/LinearSolverFactory.hpp>
#include <opm/core/utility/ErrorMacros.hpp>
#include <opm/autodiff/AutoDiffBlock.hpp>
#include <opm/autodiff/AutoDiffHelpers.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cmath>
#include <string>
#include <iomanip>

#include <Eigen/Sparse>

#include "common.hpp"




double norm(const ADB& x)
{
    return x.value().matrix().norm();
}




double euclidean_diff(const double* v0, const double* v1, const int dim)
{
    double l2 = 0.0;
    for (int dd = 0; dd < dim; ++dd) {
        const double diff = v0[dd] - v1[dd];
        l2 += diff*diff;
    }
    return std::sqrt(l2);
}




// Solving residual.derivative() * du = residual, for du. Returning u-du.
// @jny Think maybe the involvement of 'u' was just obfuscating stuff...

ADB old_solve(const Opm::LinearSolverInterface& linsolver, const ADB& residual, const ADB& u)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr = residual.derivative()[0];

    V du = V::Zero(residual.size());

    // solve(n, # nonzero values ("val"), ptr to col indices ("col_ind"), ptr to row locations in val array ("row_ind")
    // (these two may be swapped, not sure about the naming convention here...), array of actual values ("val")
    // (I guess... '*sa'...), rhs, solution)
    Opm::LinearSolverInterface::LinearSolverReport rep
        = linsolver.solve(matr.rows(), matr.nonZeros(),
                          matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                          residual.value().data(), du.data());
    if (!rep.converged) {
        THROW("Linear solver convergence failure.");
    }
    return u - du;
}




// Solving residual.derivative() * du = residual, for du. Returning u-du.

V solve(const Opm::LinearSolverInterface& linsolver, const ADB& residual)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> matr = residual.derivative()[0];

    V du = V::Zero(residual.size());

    // solve(n, # nonzero values ("val"), ptr to col indices ("col_ind"), ptr to row locations in val array ("row_ind")
    // (these two may be swapped, not sure about the naming convention here...), array of actual values ("val")
    // (I guess... '*sa'...), rhs, solution)
    Opm::LinearSolverInterface::LinearSolverReport rep
        = linsolver.solve(matr.rows(), matr.nonZeros(),
                          matr.outerIndexPtr(), matr.innerIndexPtr(), matr.valuePtr(),
                          residual.value().data(), du.data());
    if (!rep.converged) {
        THROW("Linear solver convergence failure.");
    }
    return du;
}




void print_vec(const std::string &tag, const ADB &u)
{
    //std::copy(u.value().data(), u.value().data() + u.size(), std::ostream_iterator<double>(std::cout, " "));

    std::cout << tag;
    for (int i=0; i<u.size(); i++) {
        std::cout << std::setw(15) << std::left << *( u.value().data() + i ) << " ";
    }
    std::cout << " \t(norm = " << norm(u) << ")" << std::endl;
}




void print_vec_V(const std::string &tag, const V &u)
{
    //std::copy(u.value().data(), u.value().data() + u.size(), std::ostream_iterator<double>(std::cout, " "));

    std::cout << tag;
    for (int i=0; i<u.size(); i++) {
        std::cout << std::setw(15) << std::left << ( u[i] ) << " ";
    }
    std::cout << std::endl;
    //std::cout << " \t(norm = " << norm(u) << ")" << std::endl;
}




// @jny What would be the appropriate way to do this?

M pick_elements(const M &in, const double elem)
{
    M result = in;
    for (int k=0; k<result.outerSize(); k++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(result, k); it; ++it) {
//            it.value();
//            it.row(); // row index
//            it.col(); // col index (here it is equal to k)
//            it.index(); // inner index, here it is equal to it.row()
//            std::cout << it.row() << " " << it.col() << " " << it.value() << std::endl;
            if ( fabs(it.value()-elem) > 1e-15 ) {
                const_cast<double &>( it.value() ) = 0.0;
            }
        }
    }
    return result;
}




















