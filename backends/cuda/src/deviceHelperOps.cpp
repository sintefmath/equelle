#include <Eigen/Sparse>
#include <Eigen/Eigen>
#include <iostream>

#include "DeviceHelperOps.hpp"
#include "CudaMatrix.hpp"

#include <opm/common/ErrorMacros.hpp>
#include <opm/grid/UnstructuredGrid.h>

using namespace equelleCUDA;


DeviceHelperOps::DeviceHelperOps( const UnstructuredGrid& grid_in )
    : initialized_(false),
      grad_(),
      div_(),
      fulldiv_(),
      num_int_faces_(-1),
      grid_(grid_in)
{
    // LEFT EMPTY
}


const CudaMatrix& DeviceHelperOps::grad() {
    if (grad_.isEmpty()) {
	std::cout << "Creating grad matrix\n";
	initGrad_();
    }
    return grad_;
}

const CudaMatrix& DeviceHelperOps::div() {
    if ( div_.isEmpty() ) {
	std::cout << "Creating div matrix\n";
	initDiv_();
    }
    return div_;
}

const CudaMatrix& DeviceHelperOps::fulldiv() {
    if ( fulldiv_.isEmpty() ) {
	std::cout << "Creating fulldiv matrix\n";
	initFulldiv_();
    }
    return fulldiv_;
}

int DeviceHelperOps::num_int_faces() {
    if ( num_int_faces_ == -1 ) {
	OPM_THROW(std::runtime_error, "num_int_faces_ not created in DeviceHelperOps!");
    }
    return num_int_faces_;
}




void DeviceHelperOps::initGrad_() {
    if ( initialized_ == false ) {
	initHost_();
    }
    grad_ = CudaMatrix(host_grad_);
}

void DeviceHelperOps::initDiv_() {
    if ( initialized_ == false ) {
	initHost_();
    }
    div_ = CudaMatrix(host_div_);
}

void DeviceHelperOps::initFulldiv_() {
    if ( initialized_ == false ) {
	initHost_();
    }
    fulldiv_ = CudaMatrix(host_fulldiv_);
}




void DeviceHelperOps::initHost_() {
    
    std::cout << "-----------------------------------------------------\n";
    std::cout << "---------- CREATING HOST HELPER OPS -----------------\n";
    std::cout << "-----------------------------------------------------\n";

    const int nc = grid_.number_of_cells;
    const int nf = grid_.number_of_faces;
    // Define some neighbourhood-derived helper arrays.
    typedef Eigen::Array<bool, Eigen::Dynamic, 1> OneColBool;
    typedef Eigen::Array<int, Eigen::Dynamic, 2, Eigen::RowMajor> TwoColInt;
    typedef Eigen::Array<bool, Eigen::Dynamic, 2, Eigen::RowMajor> TwoColBool;
    TwoColInt nb = Eigen::Map<TwoColInt>(grid_.face_cells, nf, 2);
    // std::cout << "nb = \n" << nb << std::endl;
    TwoColBool nbib = nb >= 0;
    OneColBool ifaces = nbib.rowwise().all();
    const int num_internal = ifaces.cast<int>().sum();
    // std::cout << num_internal << " internal faces." << std::endl;
    TwoColInt nbi(num_internal, 2);
    host_internal_faces_.resize(num_internal);
    int fi = 0;
    for (int f = 0; f < nf; ++f) {
	if (ifaces[f]) {
	    host_internal_faces_[fi] = f;
	    nbi.row(fi) = nb.row(f);
	    ++fi;
	}
    }
    // std::cout << "nbi = \n" << nbi << std::endl;
    // Create matrices.
    host_ngrad_.resize(num_internal, nc);
    host_caver_.resize(num_internal, nc);
    typedef Eigen::Triplet<double> Tri;
    std::vector<Tri> ngrad_tri;
    std::vector<Tri> caver_tri;
    ngrad_tri.reserve(2*num_internal);
    caver_tri.reserve(2*num_internal);
    for (int i = 0; i < num_internal; ++i) {
	ngrad_tri.emplace_back(i, nbi(i,0), 1.0);
	ngrad_tri.emplace_back(i, nbi(i,1), -1.0);
	caver_tri.emplace_back(i, nbi(i,0), 0.5);
	caver_tri.emplace_back(i, nbi(i,1), 0.5);
    }
    host_ngrad_.setFromTriplets(ngrad_tri.begin(), ngrad_tri.end());
    host_caver_.setFromTriplets(caver_tri.begin(), caver_tri.end());
    host_grad_ = -host_ngrad_;
    host_div_ = host_ngrad_.transpose();
    std::vector<Tri> fullngrad_tri;
    fullngrad_tri.reserve(2*nf);
    for (int i = 0; i < nf; ++i) {
	if (nb(i,0) >= 0) {
	    fullngrad_tri.emplace_back(i, nb(i,0), 1.0);
	}
	if (nb(i,1) >= 0) {
	    fullngrad_tri.emplace_back(i, nb(i,1), -1.0);
	}
    }
    host_fullngrad_.resize(nf, nc);
    host_fullngrad_.setFromTriplets(fullngrad_tri.begin(), fullngrad_tri.end());
    host_fulldiv_ = host_fullngrad_.transpose();
    
    num_int_faces_ = host_internal_faces_.rows();

    initialized_ = true;
}

