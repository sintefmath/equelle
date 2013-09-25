#pragma once

#include <string>

enum GridMapping {
	GRID_MAPPING_ENTITY, //Scalar, vector, (any non-collection)
	GRID_MAPPING_INTERIORCELLS,
	GRID_MAPPING_BOUNDARYCELLS,
	GRID_MAPPING_ALLCELLS,
	GRID_MAPPING_INTERIORFACES,
	GRID_MAPPING_BOUNDARYFACES,
	GRID_MAPPING_ALLFACES,
	GRID_MAPPING_INTERIOREDGES,
	GRID_MAPPING_BOUNDARYEDGES,
	GRID_MAPPING_ALLEDGES,
	GRID_MAPPING_INTERIORVERTICES,
	GRID_MAPPING_BOUNDARYVERTICES,
	GRID_MAPPING_ALLVERTICES,
	GRID_MAPPING_ANY, // the default length of a collection, if it is not explicitly specified
	GRID_MAPPING_INVALID, //Invalid size... should be caught by error checking
};

enum EntityType {
	TYPE_SCALAR,
	TYPE_SCALAR_AD,
	TYPE_VECTOR,
	TYPE_VERTEX,
	TYPE_EDGE,
	TYPE_FACE,
	TYPE_CELL,
	TYPE_BOOLEAN,
	TYPE_INVALID,
};

struct VariableType {
	VariableType() : entity_type(TYPE_INVALID), collection(false) {}
	EntityType entity_type;
	bool collection;

	VariableType& operator=(VariableType other) {
		entity_type = other.entity_type;
		collection = other.collection;

		return *this;
	}
};

bool operator==(const VariableType& a, const VariableType& b);
bool operator!=(const VariableType& a, const VariableType& b);


struct info
{
	info() : str(""), error_str(""), grid_mapping(GRID_MAPPING_INVALID), array_size(-1), type() {}

	info* clone() {
		//Allocate memory
		info* retval = new info();

		//copy
		*retval = *this;

		//Return
		return retval;
	}

	info& operator=(info other) {
		str = other.str;
		error_str = other.error_str;
		grid_mapping = other.grid_mapping;
		array_size = other.array_size;
		type = other.type;

		return *this;
	}

	std::string str;            // The string which is to be outputted in the C++ file
	std::string error_str;      // All errors go here; (test if (error != NULL) std::cout << error << std::endl;
	GridMapping grid_mapping;  // This defines the mapping of the variable (one value per face, cell, interior face, etc.)
	int array_size;       // The number of elements in a vector / array
	VariableType type;    // The type of the variable
};

/*
char[][] error_msgs = {
"you forgot something",
"Types do not match",
..
}
std::cout << "Error message " << inf.error_code << error_msgs[inf.error_code] << std::endl;
*/





// global structure and counter for storing the names of the variables of each type (used for stopping variables reassignment)
struct VariableStructureForCPP
{
	std::string name;           // must begin with a small letter
	VariableType type;           // can be: scalar, vector, vertex, scalars etc.
	GridMapping grid_mapping; // if the type is a singular type, then the length is 1; otherwise it can be any other number >= 1
	int array_size;         // if the variable is assigned to a list (vector, tuple etc)
	bool assigned;         // we want to know if a variable has been assigned, in order to prevent errors (example: operations with unassigned variables)
};



// global structure and counter for storing the names of the functions
struct FunctionStructureForCPP
{
	std::string name;                                      // name of the function (ex: g1)
	VariableType type;                                // type of the return (ex: type.entity_type = TYPE_SCALAR && type.collection = true)
	GridMapping grid_mapping;                         // grid mapping of the return (ex: GRID_MAPPING_ALLCELLS)
	std::string paramList;                                 // the list of parameters' types (ex: (Cell, Face, CollOfVectors, CollOfScalars On AllFaces(Grid)))
	VariableStructureForCPP headerVariables[100];     // the header variables of the function (ex: c1, f1, pv1, ps1)
	int noParam;                                      // the number of the function's parameters (ex: 4)
	VariableStructureForCPP localVariables[100];      // the local variables of the function (ex: var1, var2, var3)
	std::string signature;                                 // the C++ code for the function's parameter list (ex: (Cell c1, Face f1, CollOfVectors pv1, CollOfScalars On AllFaces(Grid) ps1))
	int noLocalVariables;                             // the number of the function's local variables (ex: 3)
	bool assigned;                                    // if the function was assigned or only declared (ex: true)
};
