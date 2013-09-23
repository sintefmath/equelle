%{
#include <string>
using namespace std;
#include "y.tab.h"

%}

%option nounput

%%
[-]         return yytext[0];
[+]         return yytext[0];
[*]         return yytext[0];
[/]         return yytext[0];
[(]         return yytext[0];
[)]         return yytext[0];
[<]         return yytext[0];
[>]         return yytext[0];
[!]         return yytext[0];
[=]         return yytext[0];
[,]         return yytext[0];
[:]         return yytext[0];
[{]         return yytext[0];
[}]         return yytext[0];
[[]			return yytext[0];
[]]			return yytext[0];
"<="        return LESSEQ;
">="        return GREATEREQ;
"=="        return EQ;
"!="        return NOTEQ;
"->"        return RET;
"::"        return HEADER_DECL;
"^"         return yytext[0];
"."         return yytext[0];
\n          return yytext[0];
"Scalar"                return SCALAR;
"Vector"                return VECTOR;
"Cell"                  return CELL;
"Face"                  return FACE;
"Edge"                  return EDGE;
"Vertex"                return VERTEX;
"ScalarAD"              return ADB;
"Boolean"               return BOOLEAN;
"Collection"            return COLLECTION;
"On"                    return ON;
"Of"                    return OF;
"Grid"                  return GRID;
"AllCells"              return ALL_CELLS;
"BoundaryCells"         return BOUNDARY_CELLS;
"InteriorCells"         return INTERIOR_CELLS;
"AllFaces"              return ALL_FACES;
"BoundaryFaces"         return BOUNDARY_FACES;
"InteriorFaces"         return INTERIOR_FACES;
"AllEdges"              return ALL_EDGES;
"BoundaryEdges"         return BOUNDARY_EDGES;
"InteriorEdges"         return INTERIOR_EDGES;
"AllVertices"           return ALL_VERTICES;
"BoundaryVertices"      return BOUNDARY_VERTICES;
"InteriorVertices"      return INTERIOR_VERTICES;
"FirstCell"             return FIRST_CELL;
"SecondCell"            return SECOND_CELL;
"Area"                  return AREA;
"Volume"                return VOLUME;
"Normal"                return NORMAL;
"Dot"                   return DOT;
"Length"                return LENGTH;
"EuclideanLength"       return EUCLIDEAN_LENGTH;
"Centroid"              return CENTROID;
"Gradient"              return GRADIENT;
"Divergence"            return DIVERGENCE;
"True"                  return TRUE;
"False"                 return FALSE;
"And"                   return AND;
"Or"                    return OR;
"Not"                   return NOT;
"Xor"                   return XOR;
"Ceil"                  return CEIL;
"Floor"                 return FLOOR;
"Abs"                   return ABS;
"Function"              return FUNCTION;
"return"                return RETURN;
"Min"                   return MIN;
"Max"                   return MAX;
"Tuple"					return TUPLE;
"UserSpecifiedScalar"				return USS;
"UserSpecifiedScalarWithDefault"	return USSWD;
"UserSpecifiedCollectionOfScalar"	return USCOS;
"Output"							return OUTPUT;



"0"                 {
	yylval.inf = new info();
	yylval.inf->str = strdup(yytext); return INTEGER;
}
[1-9][0-9]*         {
	yylval.inf = new info();
	yylval.inf->str = strdup(yytext); return INTEGER;
}
[a-z][a-zA-Z0-9_]*  {
	yylval.inf = new info();
	yylval.inf->str = strdup(yytext); return VARIABLE;
}

[#][^\n]*           {
	yylval.inf = new info();
	yylval.inf->str = strdup(yytext); return COMMENT;
}
[ \t]
.                   printf("Character %c is unknown\n", yytext[0]);


%%
int yywrap()
{
return 1;
}
