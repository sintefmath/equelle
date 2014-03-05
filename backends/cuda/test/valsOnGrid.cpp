
#include <iostream>

#include <vector>

#include <opm/core/utility/parameters/ParameterGroup.hpp>
#include <opm/core/grid.h>
#include <opm/core/grid/GridManager.hpp>
#include <opm/core/utility/ErrorMacros.hpp>

#include "EquelleRuntimeCUDA.hpp"
#include "DeviceGrid.hpp"
#include "CollOfIndices.hpp"
#include "CollOfScalar.hpp"


using namespace equelleCUDA;



int compare(CollOfScalar scal, double sol[], 
	    int sol_size,
	    std::string test);
int collOfScalarTest(EquelleRuntimeCUDA* er);

int inputDomainTest(EquelleRuntimeCUDA* er);
int inputVectorComp(std::vector<int> host, int ans[], int ans_size, std::string test);

int scalar_test(EquelleRuntimeCUDA* er);



//test_suite* init_unit_test_suite( int argc, char* argv[] )
//{
int main( int argc, char** argv) {
    
    Opm::parameter::ParameterGroup param( argc, argv, false);
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
    if ( compare(faces, faces_sol, faces_size, "inputaCollectionOfScalar(faces)") ) {
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


    return 0;
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
	    if (host[i] != sol[i]) {
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
    CollOfFace in_face = er->inputDomainSubsetOf("ind", er->allFaces());
    if ( inputVectorComp(in_face.stdToHost(), ans, 3,  "allFaces()") ) {
	return 1;
    }
    CollOfCell in_cell = er->inputDomainSubsetOf("ind", er->allCells());
    if ( inputVectorComp(in_cell.stdToHost(), ans, 3,  "allCells()") ) {
	return 1;
    }

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
    if ( host.size() != ans_size) {
	std::cout << "Error in valsOnGrid.cpp - testing inputDomainSubsetOf(" << test << ")\n";
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
