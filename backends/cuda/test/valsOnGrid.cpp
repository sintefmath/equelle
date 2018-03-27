
/* This file implements test for the following in given order:
   
   Since this file has become so large, here is a list of its content:
// - main
// - collOfScalarTest(&er): 
        - Read input from file
	- Extend test
	- On
	- Extend (subset -> subset)
	- On (subset -> subset)
	- Extend (from scalar)
	- Volume / area
	- Trinary if
	- Gradient and Divergence
	- Array of collections
	- mutable (copy assignment)
// - vector_test(&er):
	- centroid
	- sqrt(collOfScalar)
	- Normals
// - scalar_test(&er):
        - inputScalarWithDefault
// - inputDomainTest(&er):
        - inputDomainSubsetOf
	- On for CollOfIndices
	- 
*/

#include <iostream>

#include <vector>
#include <math.h>
#include <limits>

#include <opm/common/utility/parameters/ParameterGroup.hpp>
#include <opm/grid/UnstructuredGrid.h>
#include <opm/grid/GridManager.hpp>
#include <opm/common/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"
#include "CollOfVector.hpp"

using namespace equelleCUDA;



int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test);
int collOfScalarTest(EquelleRuntimeCUDA* er);

int inputDomainTest(EquelleRuntimeCUDA* er);
int inputVectorComp(std::vector<int> host, int ans[], int ans_size, std::string test);

int scalar_test(EquelleRuntimeCUDA* er);

int vector_test(EquelleRuntimeCUDA* er);
double compNorm(double a, double b, double c);


//test_suite* init_unit_test_suite( int argc, char* argv[] )
//{
int main( int argc, char** argv) {
    
    Opm::ParameterGroup param( argc, argv, false);
    EquelleRuntimeCUDA er(param);
    
    DeviceGrid dg(er.getGrid());
    
    if ( inputDomainTest(&er) ) {
        return 1;
    }

    if ( collOfScalarTest(&er) ) {
	return 1;
    }
    
    if ( scalar_test(&er) ) {
     	return 1;
    }

    if ( vector_test(&er) ) {
	return 1;
    }

    return 0;
}



int collOfScalarTest(EquelleRuntimeCUDA* er) {
    
    CollOfScalar a = er->inputCollectionOfScalar("a", er->allCells());
    double a_full_sol[] = {0,10,20,30,40,50,60,70,80,90,100,110};
    int a_full_size = 12;
    if ( compare(a, a_full_sol, a_full_size, "inputCollectionOfScalar(a)") ) {
	return 1;
    }

    CollOfScalar b = er->inputCollectionOfScalar("b", er->allCells());
    double b_full_sol[] = {124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124, 124.124};
    int b_full_size = 12;
    if ( compare(b, b_full_sol, b_full_size, "inputCollectionOfScalar(b)") ) {
	return 1;
    }

    CollOfScalar faces = er->inputCollectionOfScalar("faces", er->allFaces());
    double faces_sol[] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    for (int i = 0; i < 31; i++) {
	faces_sol[i] *= i*10;
    }
    int faces_size = 31;
    if ( compare(faces, faces_sol, faces_size, "inputCollectionOfScalar(faces)") ) {
	return 1;
    }

    // EXTEND TEST:
    
    CollOfScalar intCell = er->inputCollectionOfScalar("intCell", er->interiorCells());
    double intCell_sol[] = {3.14, 1002.2001};
    if ( compare(intCell, intCell_sol, 2, "inputCollectionOfScalar(interiorCells())") ) {
	return 1;
    }
    CollOfScalar ext_intCell_allCell = er->operatorExtend(intCell,
							  er->interiorCells(),
							  er->allCells());
    double ext_intCell_allCell_sol[] = {0,0,0,0,0,3.14,1002.2001,0,0,0,0,0};
    if ( compare(ext_intCell_allCell, ext_intCell_allCell_sol, 12,
		 "Extend(intCell, interiorCells(), allCells())") ) {
	return 1;
    }

    // Extend intFaces -> allFaces
    CollOfScalar intFace = er->inputCollectionOfScalar("intFace", er->interiorFaces());
    double intFace_sol[] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10, 11.11,
			    12.12, 13.13, 14.14, 15.15, 16.16, 17.17};
    if ( compare(intFace, intFace_sol, 17, "inputCollectionOfScalar(interiorFaces())")) {
	return 1;
    }
    CollOfScalar ext_intFace_allFace = er->operatorExtend( intFace,
							   er->interiorFaces(),
							   er->allFaces());
    double ext_intFace_allFace_sol[] = {0, 1.1, 2.2, 3.3, 0, 0, 4.4, 5.5, 6.6, 0, 0,
					7.7, 8.8, 9.9, 0, 0,0,0,0, 10.10, 11.11, 12.12,
					13.13, 14.14, 15.15, 16.16, 17.17, 0,0,0,0};
    if( compare(ext_intFace_allFace, ext_intFace_allFace_sol, 31,
		"Extend(intFace, interiorFaces(), allFaces())") ) {
	return 1;
    }

    // ON TEST
    
    // On allCells -> interiorCells
    // a is {0 10 20 ... 110} for allCells
    CollOfScalar on_allCell_intCell = er->operatorOn( a, er->allCells(),
							er->interiorCells());
    double on_allCell_intCell_sol[] = {50, 60};
    if( compare(on_allCell_intCell, on_allCell_intCell_sol, 2, 
		"On(a, allCells(), interiorCells())") ) {
	return 1;
    }

    // EXTEND subset -> subset

    // Get a subset of both interior_faces and boundary_cells (1,2,3,7,8,11)
    // This will be the subset of subset
    int subset_ans[] = {1,2,3,7,8,11};
    CollOfFace subset_face = er->inputDomainSubsetOf("subset", er->interiorFaces());
    if ( inputVectorComp(subset_face.stdToHost(), subset_ans, 6, "interiorFaces()") ) {
	return 1;
    }
    CollOfCell subset_cell = er->inputDomainSubsetOf("subset", er->boundaryCells());
    if ( inputVectorComp(subset_cell.stdToHost(), subset_ans, 6, "boundaryCells()") ) {
	return 1;
    }
    // Get input on the subset
    CollOfScalar subsetVals = er->inputCollectionOfScalar("subsetVals", subset_face);
    double subsetVals_sol[] = {10, 20, 30, 70, 80, 110};
    if ( compare(subsetVals, subsetVals_sol, 6, "input(subsetVals)") ) {
	return 1;
    }
    CollOfScalar extendSubsetCell = er->operatorExtend( subsetVals,
							subset_cell,
							er->boundaryCells());
    double extendSubsetCell_sol[] = {0, 10,20,30, 0, 70,80,0,0,110};
    if ( compare(extendSubsetCell, extendSubsetCell_sol, 10,
		 "Extend(subset_Cell -> boundaryCells())") ) {
	return 1;
    }
    CollOfScalar extendSubsetFace = er->operatorExtend( subsetVals,
							subset_face,
							er->interiorFaces());
    double extendSubsetFace_sol[] = {10,20,30,0,70,80,110,0,0,0,0,0,0,0,0,0,0};
    if ( compare(extendSubsetFace, extendSubsetFace_sol, 17,
		 "Extend(subset_Face -> interiorFaces())") ) {
	return 1;
    }

    // On subset -> subset
    // Get values on boundary_cells:
    CollOfScalar boundVals = er->inputCollectionOfScalar("boundCellVals", er->boundaryCells());
    double boundVals_sol[] = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
    if (compare(boundVals, boundVals_sol, 10, "input(boundCellVals)") ) {
	return 1;
    }
    CollOfScalar onSubsetCell = er->operatorOn( boundVals,
						er->boundaryCells(),
						subset_cell);
    double onSubsetCell_sol[] = {0.1, 0.2, 0.3, 0.5, 0.6, 0.9};
    if ( compare(onSubsetCell, onSubsetCell_sol, 6, "On(boundaryCells() -> subset_cell)")){
	return 1;
    }
    
    // Test for extend from scalar
    CollOfScalar fromScalar = er->operatorExtend(9.124, er->interiorCells());
    double fromScalar_sol[] = {9.124, 9.124};
    if ( compare(fromScalar, fromScalar_sol, 2, "Extend(9.124, InteriorCells())") ) {
	return 1;
    }

    // TEST VOLUME AND AREA
    CollOfScalar vol = er->norm(er->allCells());
    double vol_sol[] = {0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15};
    if ( compare(vol, vol_sol, 12, "Norm(AllCells())") ) {
	return 1;
    }
    CollOfScalar area = er->norm(er->allFaces());
    double x = 0.3;
    double y = 0.5;
    double area_sol[] = {x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y};
    if ( compare(area, area_sol, 31, "Norm(AllFaces())") ) {
	return 1;
    }
    CollOfScalar area_bnd = er->norm(er->boundaryFaces());
    double area_bnd_sol[] = {x,x,x,x,x,x,y,y,y,y,y,y,y,y};
    if ( compare(area_bnd, area_bnd_sol, 14, "Norm(BoundaryFaces())") ) {
	return 1;
    }
	
 
    // Test trinaryIf
    // a = {0, 10, 20, ..., 110} allCells
    // a_big = a > 50 ? a * 2 : a / 2;
    CollOfScalar a_big = er->trinaryIf( (a > er->operatorExtend(50, er->allCells())), a*2, a/2);
    double a_big_sol[] = {0, 5, 10, 15, 20, 25, 120, 140, 160, 180, 200, 220};
    if ( compare(a_big, a_big_sol, 12, "trinaryIf(a > 50 Extend AllCells(), a*2, a/2)")) {
	return 1;
    }
    
    CollOfScalar a_big_grad = er->gradient(a_big);
    double a_big_grad_sol[] = {5,5,5, 5,95,20, 20,20,20, 20,20,110,125, 140,155,80,80};
    if ( compare(a_big_grad, a_big_grad_sol, 17, "Gradient(a_big)") ) {
	return 1;
    }

    CollOfScalar div = er->divergence(a_big_grad);
    double div_sol[] = {25,20,110,120, 125,225,-105,-65, -120,-155,-80,-100};
    if ( compare(div, div_sol, 12, "Divergence(a_big_grad)") ) {
	return 1;
    }


    // Make an array of collections:
    std::tuple<CollOfScalar, CollOfScalar, CollOfScalar> myArray = makeArray(a_big, a_big_grad, div);
    if ( compare( std::get<0>(myArray), a_big_sol, 12, "myArray[0] (a_big)")) {
	return 1;
    }
    if ( compare( std::get<1>(myArray), a_big_grad_sol, 17, "myArray[1] (a_big_grad)")) {
	return 1;
    }
    if ( compare( std::get<2>(myArray), div_sol, 12, "myArray[2] (div)") ) {
	return 1;
    }

   // Try to make div be equal to a_big
    div = a_big;
    if ( compare(div, a_big_sol, 12, "div = a_big")) {
	return 1;
    }

    return 0;
}



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

    CollOfVector centrCells = er->centroid(er->allCells());
    CollOfScalar centrCells0 = centrCells[0];
    CollOfScalar centrCells1 = centrCells[1];
    double centrCells0_sol[] = {0.25,0.75,1.25,1.75, 0.25,0.75,1.25,1.75, 0.25,0.75,1.25,1.75};
    double centrCells1_sol[] = {0.15,0.15,0.15,0.15, 0.45,0.45,0.45,0.45, 0.75,0.75,0.75,0.75};
    if ( compare( centrCells0, centrCells0_sol, 12, "Centroid(AllCells)[0]") ) {
	return 1;
    }
    if ( compare( centrCells1, centrCells1_sol, 12, "Centroid(AllCells)[1]") ) {
	return 1;
    }

    CollOfVector centr_bnd = er->centroid(er->boundaryCells());
    CollOfScalar centr_bnd0 = centr_bnd[0];
    CollOfScalar centr_bnd1 = centr_bnd[1];
    double centr_bnd0_sol[] = {0.25,0.75,1.25,1.75, 0.25,1.75, 0.25,0.75,1.25,1.75};
    double centr_bnd1_sol[] = {0.15,0.15,0.15,0.15, 0.45,0.45, 0.75,0.75,0.75,0.75};
    if ( compare( centr_bnd0, centr_bnd0_sol, 10, "Centroid(boundaryCells())[0]") ) {
	return 1;
    }
    if ( compare( centr_bnd1, centr_bnd1_sol, 10, "Centroid(boundaryCells())[1]") ) {
	return 1;
    }

    CollOfVector centr_intface = er->centroid(er->interiorFaces());
    CollOfScalar centr_intface0 = centr_intface[0];
    CollOfScalar centr_intface1 = centr_intface[1];
    double centr_intface0_sol[] = {0.5,1.0,1.5, 0.5,1.0,1.5, 0.5,1.0,1.5,
				   0.25,0.75,1.25,1.75, 0.25,0.75,1.25,1.75};
    double centr_intface1_sol[] = {0.15,0.15,0.15, 0.45,0.45,0.45, 0.75,0.75,0.75,
				   0.3,0.3,0.3,0.3, 0.6,0.6,0.6,0.6};
    if ( compare( centr_intface0, centr_intface0_sol, 17, "centroid(interiorFaces)[0]")){
	return 1;
    }
    if ( compare( centr_intface1, centr_intface1_sol, 17, "centroid(interiorFaces)[1]")){
	return 1;
    }
				   

    // TESTING SQRT
    // using vec2 and sol2[] = {2,5,8,11,14,17,20,23,26,29,32,35,38,41};
    CollOfScalar sqrt_vec2 = er->sqrt(vec2);
    double sqrt_vec2_sol[14];
    for ( int i = 0; i < 14; ++i) {
	sqrt_vec2_sol[i] = sqrt(sol2[i]);
    }
    if ( compare( sqrt_vec2, sqrt_vec2_sol, 14, "er->sqrt(vec2) where vec2 is CollOfScalar")) {
	return 1;
    }

    // Testing Normals On AllFaces
    CollOfVector normalAll = er->normal(er->allFaces());
    double normalAll0_sol[] = {0.3,0.3,0.3,0.3,0.3, 0.3,0.3,0.3,0.3,0.3, 0.3,0.3,0.3,0.3,0.3, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};
    double normalAll1_sol[] = {0,0,0,0,0, 0,0,0,0,0, 0,0,0,0,0, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5};
    if ( compare( normalAll[0], normalAll0_sol, 31, "(normal(allFaces()))[0]") ) {
	return 1;
    }
    if (compare( normalAll[1], normalAll1_sol, 31, "(normal(allFaces()))[1]") ) {
	return 1;
    }

    // Testing Normals On interiorFaces
    CollOfVector normals = er->normal(er->interiorFaces());
    CollOfScalar normals0 = normals[0];
    CollOfScalar normals1 = normals[1];
    double normals0_sol[] = {0.3,0.3,0.3, 0.3,0.3,0.3, 0.3,0.3,0.3, 0,0,0,0, 0,0,0,0};
    double normals1_sol[] = {0,0,0, 0,0,0, 0,0,0, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5,0.5};
    if ( compare( normals0, normals0_sol, 17, "(normal(interiorFaces()))[0]") ) {
	return 1;
    }
    if ( compare( normals1, normals1_sol, 17, "(normal(interiorFaces()))[1]") ) {
	return 1;
    }

    return 0;
} // Testing vectors

double compNorm(double a, double b, double c) {
    return sqrt( a*a + b*b + c*c);
}


int scalar_test(EquelleRuntimeCUDA* er) {
    double scal_1 = er->inputScalarWithDefault("scal1", 3.14);
    if ( scal_1 != 2.7182 ) {
	std::cout << "Error in valsOnGrid.cpp - testing inputScalarWithDefault\n";
	std::cout << "\tShould find value from file.\n";
	std::cout << "\t scal_1 is " << scal_1 << " but should be 2.7182\n";
	return 1;
    }
    double scal_2 = er->inputScalarWithDefault("scal2", 164.93032);
    if ( scal_2 != 164.93032) {
	std::cout << "Error in valsOnGrid.cpp - testing inputScalarWithDefault\n";
	std::cout << "\tShould take default value 164.93032 but scal_2 is " << scal_2 << std::endl;
	return 1;
    }

    return 0;
}

int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test) 
{ 
    std::cout << "\nTesting " << test << "\n";
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


int inputDomainTest(EquelleRuntimeCUDA* er) {

    int ans[] = {0,5,10};
    CollOfFace in_bnd = er->inputDomainSubsetOf("ind", er->boundaryFaces());
    if ( inputVectorComp(in_bnd.stdToHost(), ans, 3,
			 "inputDomainSubsetOf(boundaryFaces())") ) {
	return 1;
    }
    CollOfCell in_cell = er->inputDomainSubsetOf("ind", er->allCells());
    if ( inputVectorComp(in_cell.stdToHost(), ans, 3,  
			 "inputDomainSubsetOf(allCells())") ) {
	return 1;
    }
    CollOfFace in_face = er->inputDomainSubsetOf("ind", er->allFaces());
    if ( inputVectorComp(in_face.stdToHost(), ans, 3,  
			 "inputDomainSubsetOf(allFaces())") ) {
	return 1;
    }
    //CollOfFace in_bnd = er->inputDomainSubsetOf("ind", er->boundaryFaces());
    //if ( inputVectorComp(in_bnd.stdToHost(), ans, 3,
    //			 "inputDomainSubsetOf(boundaryFaces())") ) {
    //	return 1;
    //}

    // Test On for CollOfIndices
    CollOfCell second_bndface = er->secondCell(er->boundaryFaces());
    int second_bndface_sol[] = {0,-1,4,-1,8,-1,0,1,2,3,-1,-1,-1,-1};
    if ( inputVectorComp(second_bndface.stdToHost(), second_bndface_sol, 14,
			 "secondCell(boundaryFaces())") ) {
	return 1;
    }
    CollOfCell on_res = er->operatorOn(second_bndface, er->boundaryFaces(), in_face);
    int on_res_sol[] = {0, 4, 8};
    if ( inputVectorComp(on_res.stdToHost(), on_res_sol, 3,
			 "On(second_bndface, BoundaryFaces(), in_face)") ) {
	return 1;
    }

    return 0;
}



int inputVectorComp(std::vector<int> host, int ans[], int ans_size, std::string test) {

    std::cout << "\nTesting " << test << "\n";

    if ( host.size() != ans_size) {
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe collection is of wrong size!\n";
	std::cout << "\tSize is " << host.size() << " should be 3\n";
	return 1;
    }

    bool correct = true;
    std::cout << "Input indices:\n";
    for( int i = 0; i < host.size(); i++) {
	std::cout << host[i] << " ";
	if ( host[i] != ans[i]) {
	    correct = false;
	}
    }
    std::cout << "\n";
    if ( !correct ) {
	std::cout << "This is wrong\n";
	std::cout << "Error in valsOnGrid.cpp - testing " << test << "\n";
	std::cout << "\tThe indices in the collection is wrong\n";
	std:: cout << "\tShould be 0 1 2\n";
	return 1;
    }
    std::cout << "\tThis is correct\n";

    return 0;
}
