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




typedef AutoDiff::ForwardBlock<double> ADB;
typedef ADB::V V;
typedef ADB::M M;
typedef Eigen::Array<double,
                     Eigen::Dynamic,
                     Eigen::Dynamic,
                     Eigen::RowMajor> DataBlock;




double norm(const ADB& x);

double euclidean_diff(const double* v0, const double* v1, const int dim);

ADB old_solve(const Opm::LinearSolverInterface& linsolver, const ADB& residual, const ADB& u);

V solve(const Opm::LinearSolverInterface& linsolver, const ADB& residual);

void print_vec(const std::string &tag, const ADB &u);
void print_vec_V(const std::string &tag, const V &u);

M pick_elements(const M &in, const double elem);
