// libraries and constants definitions

#include<stdio.h>
#include<stdlib.h>
#define _USE_MATH_DEFINES
#include<math.h>

const int number_faces = 10;
const int number_cells = 3;
const int dimension_number = 3;
const double u[number_cells] = {1.5, 2.5, 3.5};
const double u0[number_cells] = {M_PI, M_SQRT2, M_LN2};
const double dt = 1;
const double k = 0.0146;	// for CO2


// these declarations will all be automatically generated in C++ by Equelle


struct Cell		// a cell is a structure with several components
{
	double centroid[dimension_number];		// each cell has a centroid (of dimension_number coordinate values)
	double volume;							// each cell has a volume (the space must have at least 3 dimensions)
};


struct Face		// a face is a structure which links 2 cells (one of them can be null, if the cell is at the margin of the grid)
{
	Cell first;		// the first cell
	Cell second;	// the second cell
	double area;	// the area of the face
};


struct Grid		// the grid is a structure composed of cells and faces
{
	Face face[number_faces];
	Cell cell[number_cells];
} grid;


bool operator ==(const Cell &lhs, const Cell &rhs)		// 2 cells are equal if they have the same centroid
{
	for(int i = 0; i < dimension_number; i++)
		if(lhs.centroid[i] != rhs.centroid[i])
			return false;
	return true;
}


bool operator ==(const Face &lhs, const Face &rhs)		// 2 faces are equal if they have the same first and second cells
{
	if(lhs.first == rhs.first && lhs.second == rhs.second)
		return true;
	return false;
}


int cell_faces_number(Cell c)		// returns the number of faces adjacent to a given cell
{
	int number = 0;
	for(int i = 0; i < number_faces; i++)
		if(grid.face[i].first == c || grid.face[i].second == c)
			number++;
	return number;
}


Face* cell_faces(Cell c)		// returns the faces adjacent to a given cell
{
	int number_of_faces = cell_faces_number(c);
	Face** faces_of_the_cell = (Face **) malloc(number_of_faces * sizeof(Face*));
	int counter = 0;
	for(int i = 0; i < number_faces; i++)
		if(grid.face[i].first == c || grid.face[i].second == c)
			*faces_of_the_cell[counter++] = grid.face[i];

	return *faces_of_the_cell;
}


double length(Face f)		// computes the length of the vector between the 2 centroids of the input face's cells
{
	double sum = 0;
	for(int i = 0; i < dimension_number; i++)
		sum += (f.first.centroid[i] - f.second.centroid[i]) * (f.first.centroid[i] - f.second.centroid[i]);

	return sqrt(sum);		// the Euclidean distance between the 2 centroids
}


Cell cell_neighbour(Face f, Cell c)		// returns the neighbour cell of the input cell connected through the input face
{
	if(f.first == c)
		return f.second;
	return f.first;
}


int cell_number(Cell c)		// returns the cell number from the array of cells which constructs the grid
{
	for(int i = 0; i < number_cells; i++)
		if(grid.cell[i] == c)
			return i;
	return -1;		// cell not found in the grid's structure
}


int face_number(Face f)		// returns the face number from the array of faces which constructs the grid
{
	for(int i = 0; i < number_faces; i++)
		if(grid.face[i] == f)
			return i;
	return -1;		// face not found in the grid's structure
}


// alternative 1


int main()
{
	// the cells and faces of the grid should be declared here


	// here starts the generated C++ code

	double centroid_distances[number_faces];
	for(int i = 0; i < number_faces; i++)
		centroid_distances[i] = length(grid.face[i]);

	double residual[number_cells];
	for(int i = 0; i < number_cells; i++)
	{
		residual[i] = u[i] - u0[i];
		for(int j = 0; j < cell_faces_number(grid.cell[i]); j++)
		{
			Cell neighbour_cell = cell_neighbour(cell_faces(grid.cell[i])[j], grid.cell[i]);
			residual[i] += (dt / grid.cell[i].volume) * k * (cell_faces(grid.cell[i])[j]).area * (u[cell_number(neighbour_cell)] - i) / residual[face_number(cell_faces(grid.cell[i])[j])];
		}
	}
}
