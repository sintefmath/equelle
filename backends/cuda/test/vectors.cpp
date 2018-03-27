
// This test covers functions found in the class CollOfVectors.
//


#include <iostream>

#include <vector>
#include <limits>
#include <string>

#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/grid/GridManager.hpp>
#include <opm/common/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "CollOfVector.hpp"


using namespace equelleCUDA;

int vector_test(EquelleRuntimeCUDA* er);
int compare(CollOfScalar scal,
	    double sol[],
	    int sol_size,
	    std::string test);
double compNorm(double a, double b, double c);




int main( int argc, char** argv) {
    Opm::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);
    
    if ( vector_test(&er) ) {
	return 1;
    }

    return 0;
}



// --------- TESTING COLL_OF_VECTOR ----------------
int vector_test(EquelleRuntimeCUDA* er) {

    // Set up a vector with 42 (14*3) elements {0,1,2,...,41}
    std::vector<double> host_vec(0);
    for( int i = 0; i < 42; ++i) {
	host_vec.push_back(i);
    }
    CollOfVector myVec(host_vec,3);
    
    double sol0[] = {0,3,6,9,12,15,18,21,24,27,30,33,36,39};
    double sol1[] = {1,4,7,10,13,16,19,22,25,28,31,34,37,40};
    double sol2[] = {2,5,8,11,14,17,20,23,26,29,32,35,38,41};
    CollOfScalar vec0 = myVec[0];
    CollOfScalar vec1 = myVec[1];
    CollOfScalar vec2 = myVec[2];
    if ( compare( vec0, sol0, 14, "myVec[0]") ) {
	return 1;
    }
    if ( compare( vec1, sol1, 14, "myVec[1]") ) {
	return 1;
    }
    if ( compare( vec2, sol2, 14, "myVec[2]") ) {
	return 1;
    }

    CollOfVector addedVec = myVec + myVec;
    CollOfScalar added1 = addedVec[1];
    double added1_sol[] = {2,8,14,20,26,32,38,44,50,56,62,68,74,80};
    if ( compare( added1, added1_sol, 14, "(myVec + myVec)[1]") ) {
	return 1;
    }

    CollOfVector subVec = myVec - addedVec;
    CollOfScalar sub2 = subVec[2];
    double sub2_sol[] = {-2,-5,-8,-11,-14,-17,-20,-23,-26,-29,-32,-35,-38,-41};
    if ( compare( sub2, sub2_sol, 14, "(myVec - addedVec)[2]") ) {
	return 1;
    }
    
    // Testing copy assignment operator
    subVec = myVec;
    if(compare( subVec[0], sol0, 14, "(subVec = myVec)[0] - copy assigment op." ) ) {
	return 1;
    }
    if(compare( subVec[1], sol1, 14, "(subVec = myVec)[1] - copy assignment op.") ) {
	return 1;
    }
    if(compare( subVec[2], sol2, 14, "(subVec = myVec)[2] - copy assignment op.") ) {
	return 1;
    }
    // subVec is the same as the given on top of the function.


    // Multiplication with scalar
    CollOfVector scalVecMult = (-2.0)*subVec;
    CollOfScalar scalVecMult1 = scalVecMult[1];
    double solScalVecMult1[] = {-2,-8,-14,-20,-26,-32,-38,-44,-50,-56,-62,-68,-74,-80};
    if ( compare ( scalVecMult1, solScalVecMult1, 14, "(-2*subVec)[1]") ) {
	return 1;
    }
    CollOfVector vecScalMult = subVec*2.02;
    CollOfScalar vecScalMult1 = vecScalMult[1];
    double solVecScalMult1[] = {2.02, 8.08, 14.14, 20.20, 26.26, 32.32, 38.38, 44.44, 50.50, 56.56, 62.62, 68.68, 74.74, 80.80};
    if ( compare ( vecScalMult1, solVecScalMult1, 14, "(subVec*2.02)[1]") ) {
	return 1;
    }
    
    // Multiplication CollOfVector and CollOfScalar
    std::vector<double> multHost = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    CollOfScalar multCos(multHost);
    CollOfVector cosMultcov = multCos * subVec;
    double cosMultcov0[14], cosMultcov1[14], cosMultcov2[14];
    for (int i = 0; i < 14; ++i) {
	cosMultcov0[i] = sol0[i]*multHost[i];
	cosMultcov1[i] = sol1[i]*multHost[i];
	cosMultcov2[i] = sol2[i]*multHost[i];
    }
    if ( compare( cosMultcov[0], cosMultcov0, 14, "(multCos*subVec)[0]") ) {
	return 1;
    }
    if ( compare( cosMultcov[1], cosMultcov1, 14, "(multCos*subVec)[1]") ) {
	return 1;
    }
    if ( compare( cosMultcov[2], cosMultcov2, 14, "(multCos*subVec)[2]") ) {
	return 1;
    }

    // Multiplication CollOfScalar and CollOfVector
    CollOfVector covMultcos = cosMultcov * multCos;
    for (int i = 0; i < 14; ++i ) {
	cosMultcov0[i] = cosMultcov0[i] * multHost[i];
	cosMultcov1[i] = cosMultcov1[i] * multHost[i];
	cosMultcov2[i] = cosMultcov2[i] * multHost[i];
    }
    if ( compare( covMultcos[0], cosMultcov0, 14, "(cosMultcov * multCos)[0]") ) {
	return 1;
    }
    if ( compare( covMultcos[1], cosMultcov1, 14, "(cosMultcov * subVec)[1]") ) {
	return 1;
    }
    if ( compare( covMultcos[2], cosMultcov2, 14, "(cosMultcov * subVec)[2]") ) {
	return 1;
    }
    
    // Division CollOfVector and CollOfScalar
    CollOfVector covDivcos = covMultcos / multCos;
    for (int i = 0; i < 14; ++i ) {
	cosMultcov0[i] = cosMultcov0[i] / multHost[i];
	cosMultcov1[i] = cosMultcov1[i] / multHost[i];
	cosMultcov2[i] = cosMultcov2[i] / multHost[i];
    }
    if ( compare( covDivcos[0], cosMultcov0, 14, "(cosMultcov / multCos)[0]") ) {
	return 1;
    }
    if ( compare( covDivcos[1], cosMultcov1, 14, "(cosMultcov / subVec)[1]") ) {
	return 1;
    }
    if ( compare( covDivcos[2], cosMultcov2, 14, "(cosMultcov / subVec)[2]") ) {
	return 1;
    }

    // Division CollOfVector and Scalar
    double scal = 0.2;
    CollOfVector covDivscal = covDivcos / scal;
    for (int i = 0; i < 14; ++i ) {
	cosMultcov0[i] /= scal;
	cosMultcov1[i] /= scal;
       	cosMultcov2[i] /= scal;
    }
    if ( compare( covDivscal[0], cosMultcov0, 14, "(cosMultcov / scal)[0]") ) {
	return 1;
    }
    if ( compare( covDivscal[1], cosMultcov1, 14, "(cosMultcov / scal)[1]") ) {
	return 1;
    }
    if ( compare( covDivscal[2], cosMultcov2, 14, "(cosMultcov / scal)[2]") ) {
	return 1;
    }

    // Unary minus
    CollOfVector subVecNeg = -subVec;
    CollOfScalar subNeg2 = subVecNeg[2];
    double subNeg2_sol[] = {-2,-5,-8,-11,-14,-17,-20,-23,-26,-29,-32,-35,-38,-41};
    if ( compare ( subNeg2, subNeg2_sol, 14, "(subVecNeg = -subVec)[2]") ) {
	return 1;
    }

    // Norm
    CollOfScalar norm = myVec.norm();
    double norm_sol[14];
    for (int i = 0; i < 14; ++i) {
	norm_sol[i] = compNorm(sol0[i], sol1[i], sol2[i]);
    }
    if ( compare( norm, norm_sol, 14, "myVec.norm()") ) {
	return 1;
    }

    // Dot
    CollOfScalar dot = er->dot(covDivscal, myVec);
    double dotSol[14];
    for (int i = 0; i < 14; i++) {
	dotSol[i] = cosMultcov0[i]*sol0[i];
	dotSol[i] += cosMultcov1[i]*sol1[i];
	dotSol[i] += cosMultcov2[i]*sol2[i];
    }
    if ( compare ( dot, dotSol, 14, "dot(covDivscal, myVec)" ) ) {
	return 1;
    }

    // End of plain vector testing. Put the above in its own test file. Start line 293.
    
    return 0;
}




int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test) 
{ 
    // Test size:
    if ( scal.size() != sol_size ) {
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe collection is of wrong size!\n";
	std::cout << "\tSize is " << scal.size() << " but should be " << sol_size << "\n";
	return 1;
    }
    
    // Testing indices
    std::vector<double> host = scal.copyToHost();
    std::cout << "CollOfScalar " << test << " is the following:\n";
    bool correct = true;
    for (int i = 0; i < host.size(); ++i) {
	std::cout << host[i] << " ";
	if (i < sol_size) {
	    //if (host[i] != sol[i]) {
	    if ( fabs(host[i] - sol[i]) > 1000*std::numeric_limits<double>::epsilon() ) {
		std::cout << "(<- " << sol[i] << ") ";
		correct = false;
	    }
	}
    }
    if (correct) {
	std::cout << "\n\tThis is correct\n";
    } else {
	std::cout << "\n\tThis is wrong\n";
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	return 1;
    }

    return 0;

}


double compNorm(double a, double b, double c) {
    return sqrt( a*a + b*b + c*c);
}
