
#ifndef EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
#define EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED


//#include <opm/autodiff/AutoDiffHelpers.hpp>

#include <Eigen/Sparse>
#include <iostream>

#include "CudaMatrix.hpp"


namespace equelleCUDA {


    typedef Eigen::SparseMatrix<Scalar> M;

    struct DeviceHelperOps {
	
	CudaMatrix grad;
	CudaMatrix div;
	CudaMatrix fulldiv;

	DeviceHelperOps( const M& hostGrad, 
			 const M& hostDiv, 
			 const M& hostFullDiv)
	    : grad(hostGrad),
	      div(hostDiv),
	      fulldiv(hostFullDiv)
	{ std::cout << "Constructor DeviceHelperOps\n"; };

	~DeviceHelperOps() { std::cout << "Destructor DeviceHelperOps\n"; };

    };


} // namespace equelleCUDA


#endif // EQUELLE_DEVICEHELPEROPS_HEADER_INCLUDED
