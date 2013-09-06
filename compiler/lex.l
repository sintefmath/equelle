%{
#include "y.tab.h"
#include "string.h"
%}

%%
[-]			return yytext[0];
[+]			return yytext[0];
[*]			return yytext[0];
[/]			return yytext[0];
[(]			return yytext[0];
[)]			return yytext[0];
[<]			return yytext[0];
[>]			return yytext[0];
[!]			return yytext[0];
[=]			return yytext[0];
[,]			return yytext[0];
[:]			return yytext[0];
[{]			return yytext[0];
[}]			return yytext[0];
"<="		return LESSEQ;
">="		return GREATEREQ;
"=="		return EQ;
"!="		return NOTEQ;
"->"		return RET;
"::"		return HEADER_DECL;
"^"			return yytext[0];
"."			return yytext[0];
\n			return yytext[0];
"Scalar"				return SCALAR;
"Vector"				return VECTOR;
"Cell"					return CELL;
"Face"					return FACE;
"Edge"					return EDGE;
"Vertex"				return VERTEX;
"ScalarAD"				return ADB;
"Boolean"				return BOOLEAN;
"Collection"			return COLLECTION;
"On"					return ON;
"Of"					return OF;
"Grid"					return GRID;
"AllCells"				return ALL_CELLS;
"BoundaryCells"			return BOUNDARY_CELLS;
"InteriorCells"			return INTERIOR_CELLS;
"AllFaces"				return ALL_FACES;
"BoundaryFaces"			return BOUNDARY_FACES;
"InteriorFaces"			return INTERIOR_FACES;
"AllEdges"				return ALL_EDGES;
"BoundaryEdges"			return BOUNDARY_EDGES;
"InteriorEdges"			return INTERIOR_EDGES;
"AllVertices"			return ALL_VERTICES;
"BoundaryVertices"		return BOUNDARY_VERTICES;
"InteriorVertices"		return INTERIOR_VERTICES;
"FirstCell"				return FIRST_CELL;
"SecondCell"			return SECOND_CELL;
"FirstCells"			return FIRST_CELLS;
"SecondCells"			return SECOND_CELLS;
"Area"					return AREA;
"Areas"					return AREAS;
"Volume"				return VOLUME;
"Volumes"				return VOLUMES;
"Normal"				return NORMAL;
"Normals"				return NORMALS;
"Dot"					return DOT;
"Dots"					return DOTS;
"Length"				return LENGTH;
"Lengths"				return LENGTHS;
"EuclideanLength"		return EUCLIDEAN_LENGTH;
"EuclideanLengths"		return EUCLIDEAN_LENGTHS;
"Centroid"				return CENTROID;
"Centroids"				return CENTROIDS;
"Gradient"				return GRADIENT;
"Gradients"				return GRADIENTS;
"Divergence"			return DIVERGENCE;
"Divergences"			return DIVERGENCES;
"True"					return TRUE;
"False"					return FALSE;
"And"					return AND;
"Or"					return OR;
"Not"					return NOT;
"Xor"					return XOR;
"Ceil"					return CEIL;
"Floor"					return FLOOR;
"Abs"					return ABS;
"Ceils"					return CEILS;
"Floors"				return FLOORS;
"Abses"					return ABSES;
"Function"				return FUNCTION;
"return"				return RETURN;
"Min"					return MIN;
"Max"					return MAX;




"0"					{yylval.str = strdup(yytext); return NUMBER;}
[1-9][0-9]*		 	{yylval.str = strdup(yytext); return NUMBER;}
[a-z][a-zA-Z0-9_]*	{yylval.str = strdup(yytext); return VARIABLE;}
  /* [g][a-zA-Z0-9_]*	{yylval.str = strdup(yytext); return FUNCTION_VARIABLE;}		*/


[#][^\n]*			{yylval.str = strdup(yytext); return COMMENT;}
[ \t]
.					printf("Character %c is unknown\n", yytext[0]);


%%
int yywrap()
{
return 1;
}
