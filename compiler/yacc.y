%token NUMBER
%token SCALAR
%token VECTOR
%token CELL
%token FACE
%token EDGE
%token VERTEX
%token ADB
%token BOOLEAN
%token COLLECTION
%token ON
%token OF
%token GRID
%token ALL_CELLS
%token BOUNDARY_CELLS
%token INTERIOR_CELLS
%token ALL_FACES
%token BOUNDARY_FACES
%token INTERIOR_FACES
%token ALL_EDGES
%token BOUNDARY_EDGES
%token INTERIOR_EDGES
%token ALL_VERTICES
%token BOUNDARY_VERTICES
%token INTERIOR_VERTICES
%token FIRST_CELL
%token SECOND_CELL
%token FIRST_CELLS
%token SECOND_CELLS
%token AREA
%token AREAS
%token VOLUME
%token VOLUMES
%token NORMAL
%token NORMALS
%token DOT
%token DOTS
%token LENGTH
%token LENGTHS
%token EUCLIDEAN_LENGTH
%token EUCLIDEAN_LENGTHS
%token CENTROID
%token CENTROIDS
%token GRADIENT
%token GRADIENTS
%token DIVERGENCE
%token DIVERGENCES
%token VARIABLE
%token FUNCTION_VARIABLE
%token TRUE
%token FALSE
%token OR
%token AND
%token NOT
%token XOR
%token CEIL
%token FLOOR
%token ABS
%token CEILS
%token FLOORS
%token ABSES
%token RETURN
%token FUNCTION
%token LESSEQ
%token GREATEREQ
%token EQ
%token NOTEQ
%token RET
%token HEADER_DECL
%token COMMENT
%token MIN
%token MAX






%start pr

%left OR
%left AND
%left XOR
%left EQ NOTEQ
%left LESSEQ GREATEREQ '<' '>'
%left '+' '-'
%left '*' '/'
%right '^'
%right NOT



%{

#include <stdio.h>
#include <string.h>
#ifndef _MSC_VER
#include <stdbool.h>
#endif

void yyerror(const char* s);
char* append1(char *s1, char s2, char *s3);
char* append2(char s1, char *s2, char s3);
char* append3(char *s1, char s2, char *s3, char s4);
char* append4(char *s1, char s2, char *s3, char s4, char *s5, char s6);
char* append5(char *s1, char s2, char *s3);
char* append6(char *s1, char *s2, char *s3);
char* append7(char s1, char *s2);
char* append8(char s1, char s2, char *s3, char s4);
char* append9(char *s1, char s2, char *s3);
char* append10(char *s1, char *s2);
char* append11(char *s1, char *s2, char s3, char *s4, char s5);
char* append12(char *s1, char s2, char *s3, char s4);
char* append13(char *s1, char *s2, char *s3, char *s4, char *s5, char *s6, char s7);
char* append14(char s1, char *s2, char s3, char s4, char s5, char *s6, char s7);
char* append15(char s1, char *s2, char s3, char *s4, char s5, char *s6, char s7);
bool find1(char *s1, char *s2);
char* find2(char *s1);
char* find3(char *s1);
int find4(char *s1);
char* find5(char *s1);
char* find6(char *s1);
bool check1(char *s1);
bool check2(char *s1);
bool check3(char *s1);
bool check4(char *s1);
bool check5(char *s1);
bool check6(char *s1);
bool check7(char *s1);
char* getType(char *s1);
int getIndex1(char *s1);
int getIndex2(char *s1);
double getSize1(char *s1);
double getSize2(char *s1);
double getSize3(char *s1);
char* extract(char *s1);
char *structureToString(char *st);
char* singular_declaration_function(char *st1, char *st2);
char* plural_declaration_function(char *st1, char *st2);
char* extended_plural_declaration_function(char *st1, char *st2, char *st3, double d1);
char* singular_assignment_function(char *st1, char *st2, char *st3, char *st4);
char* plural_assignment_function(char *st1, char *st2, char *st3, char *st4, double d1);
char* singular_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4);
char* plural_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4, double d1);
char* extended_plural_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4, char *st5, double d1, double d2);





// global structure and counter for storing the names of the variables of each type (used for stopping variables reassignment)
struct VariableStructure
{
  char *name;       // must begin with a small letter
  char *type;       // can be: scalar, vector, vertex, scalars etc.
  double length;    // if the type is a singular type, then the length is 1; otherwise it can be any other number >= 1
  bool assigned;    // we want to know if a variable has been assigned, in order to prevent errors (example: operations with unassigned variables)
}
var[10000];

int varNo = 0;


// global structure and counter for storing the names of the functions
struct FunctionStructure
{
  char *name;                 // g1
  char *returnType;           // Collection Of Scalars
  double returnSize;          // 8
  char *paramList;            // (Cell, Face, Collection Of Vectors, Collection Of Scalars On AllFaces(Grid))
  struct VariableStructure headerVariables[100];     // (c1, f1, pv1, ps1)
  int noParam;                // 4
  struct VariableStructure localVariables[100];      // var1, var2, var3
  int noLocalVariables;       // 3
  bool assigned;              // false
}
fun[10000];

int funNo = 0;


bool insideFunction = false;
int currentFunctionIndex = -1;



int currentLineNumber = 1;


/*
For a d-dimensional s-sized uniform hypercube grid, the values are:
(s-2)^d                          internal cells
s^d - (s-2)^d                    boundary cells
s^d                              all cells
(d+1)d^s - (d+1)d^s + 2ds^(d-1)  internal faces
(d+1)d^s - 2ds^(d-1)             boundary faces
(d+1)d^s                         all faces
...
*/
// we define constant values for the number of cells, faces, edges and vertices in the grid, in order to detect length-mismatch errors at operations involving these entities
// as they don't collide (for example, an operation to involve both collections on internal cells and on boundary cells), we need to define these constant values to be coprime and distanced integers (in order to avoid creating a relation between them)

#define INTERIORCELLS     1.01
#define BOUNDARYCELLS     1.02
#define ALLCELLS          1.03
#define INTERIORFACES     1.04
#define BOUNDARYFACES     1.05
#define ALLFACES          1.06
#define INTERIOREDGES     1.07
#define BOUNDARYEDGES     1.08
#define ALLEDGES          1.09
#define INTERIORVERTICES  1.10
#define BOUNDARYVERTICES  1.11
#define ALLVERTICES       1.12
#define ANY               1.13      // the default length of a collection, if it is not explicitly specified

%}

%type<str> scalar_expr
%type<inf> scalar_exprs
%type<str> scalar_term
%type<inf> scalar_terms
%type<str> scalar_factor
%type<inf> scalar_factors
%type<str> numbers
%type<inf> scalars
%type<str> vector_expr
%type<str> vector_term
%type<inf> vectors
%type<inf> vector_exprs
%type<inf> vector_terms
%type<str> vertex
%type<inf> vertices
%type<str> edge
%type<inf> edges
%type<str> face
%type<inf> faces
%type<str> cell
%type<inf> cells
%type<str> adb
%type<inf> adbs
%type<str> boolean_expr
%type<str> boolean_term
%type<inf> boolean_exprs
%type<inf> boolean_terms
%type<str> NUMBER
%type<str> VARIABLE
%type<str> FUNCTION_VARIABLE
%type<str> COMMENT
%type<inf> plural
// %type<str> header
// %type<str> parameter_list
// %type<inf> type
// %type<str> end_lines
// %type<inf> return_instr
// %type<str> function_start
// %type<str> function_declaration
// %type<str> function_assignment
// %type<str> commands
%type<str> command
%type<str> command2
%type<str> singular_declaration
%type<str> plural_declaration
%type<str> extended_plural_declaration
%type<str> singular_assignment
%type<str> plural_assignment
%type<str> singular_declaration_with_assignment
%type<str> plural_declaration_with_assignment
%type<str> extended_plural_declaration_with_assignment
%type<str> declaration
%type<str> assignment
%type<str> declaration_with_assignment





%code requires
{
  struct info
  {
      double size;
      char *str;
  };
}



%union
{
  int value;
  char *str;          // the non-terminals which need to store only the translation code for C++ will be declared with this type
  struct info inf;    // the non-terminals which need to store both the translation code for C++ and the size of the collection will be declared with this type
};


%%


scalar_expr: scalar_term 		                 {$$ = strdup($1);}
           | '-' scalar_term                 {char *str = append7('-',$2); $$ = strdup(str); free(str);}
           | scalar_expr '+' scalar_term	   {char *str = append1($1,'+',$3); $$ = strdup(str); free(str);}
           | scalar_expr '-' scalar_term	   {char *str = append1($1,'-',$3); $$ = strdup(str); free(str);}
           ;


scalar_term: scalar_factor		                       {$$ = strdup($1);}
           | scalar_term '*' scalar_factor	         {char *str = append1($1,'*',$3); $$ = strdup(str); free(str);}
           | scalar_factor '*' scalar_term           {char *str = append1($1,'*',$3); $$ = strdup(str); free(str);}
           | scalar_term '/' scalar_factor	         {char *str = append1($1,'/',$3); $$ = strdup(str); free(str);}
           | scalar_term '^' scalar_factor           {char *str = append4("er.pow", '(', $1, ',', $3, ')'); $$ = strdup(str); free(str);}
           ;


scalar_factor: NUMBER		                               {$$ = strdup($1);}
             | NUMBER '.' NUMBER                       {char *str = append9($1, '.', $3); $$ = strdup(str); free(str);}
             | '(' scalar_expr ')'	                   {char *str = append2('(', $2, ')'); $$ = strdup(str); free(str);}
             | EUCLIDEAN_LENGTH '(' vector_expr ')'    {char *str = append3("er.euclideanLength", '(', $3, ')'); $$ = strdup(str); free(str);}
             | LENGTH '(' edge ')'                     {char *str = append3("er.length", '(', $3, ')'); $$ = strdup(str); free(str);}
             | AREA '(' face ')'                       {char *str = append3("er.area", '(', $3, ')'); $$ = strdup(str); free(str);}
             | VOLUME '(' cell ')'                     {char *str = append3("er.volume", '(', $3, ')'); $$ = strdup(str); free(str);}
             | DOT '(' vector_expr ',' vector_expr ')' {char *str = append4("er.dot", '(', $3, ',', $5, ')'); $$ = strdup(str); free(str);}
             | CEIL '(' scalar_expr ')'                {char *str = append3("er.ceil", '(', $3, ')'); $$ = strdup(str); free(str);}
             | FLOOR '(' scalar_expr ')'               {char *str = append3("er.floor", '(', $3, ')'); $$ = strdup(str); free(str);}
             | ABS '(' scalar_expr ')'                 {char *str = append3("er.abs", '(', $3, ')'); $$ = strdup(str); free(str);}
             | MIN '(' scalars ')'                     {char *str = append3("er.min", '(', $3.str, ')'); $$ = strdup(str); free(str);}
             | MAX '(' scalars ')'                     {char *str = append3("er.max", '(', $3.str, ')'); $$ = strdup(str); free(str);}
             | VARIABLE                                {
                                                          if(strcmp(getType($1), "scalar") != 0)
                                                          {
                                                              $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                              $$ = strcat($$, "  ");
                                                              $$ = strcat($$, $1);
                                                          }
                                                          else
                                                          {
                                                              $$ = strdup($1);
                                                          }
                                                       }
             ;


scalars: scalar_exprs                 {$$.str = strdup($1.str); $$.size = $1.size;}
       | scalars ',' scalar_exprs     {char *str = append5($1.str,',',$3.str); $$.str = strdup(str); free(str); $$.size = $1.size + $3.size;}
       | scalar_expr                  {$$.str = strdup($1); $$.size = 1;}
       | scalars ',' scalar_expr      {char *str = append5($1.str,',',$3); $$.str = strdup(str); free(str); $$.size = $1.size + 1;}
       ;


scalar_exprs: scalar_terms                     {$$.str = strdup($1.str); $$.size = $1.size;}
            | '-' scalar_terms                 {char *str = append7('-',$2.str); $$.str = strdup(str); free(str); $$.size = $2.size;}
            | scalar_exprs '+' scalar_terms
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                      $$.str = strdup("length_mismatch_error");
                                                  else
                                                  {
                                                      char *str = append1($1.str,'+',$3.str);
                                                      $$.str = strdup(str);
                                                      free(str);
                                                      $$.size = $1.size;
                                                  }
                                               }

            | scalar_exprs '-' scalar_terms
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                      $$.str = strdup("length_mismatch_error");
                                                  else
                                                  {
                                                      char *str = append1($1.str,'-',$3.str);
                                                      $$.str = strdup(str);
                                                      free(str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            ;


scalar_terms: scalar_factors                    {$$.str = strdup($1.str); $$.size = $1.size;}
            | scalar_terms '*' scalar_factors
                                                {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                      $$.str = strdup("length_mismatch_error");
                                                  else
                                                  {
                                                      char *str = append1($1.str,'*',$3.str);
                                                      $$.str = strdup(str);
                                                      free(str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_factors '*' scalar_terms
                                                {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                      $$.str = strdup("length_mismatch_error");
                                                  else
                                                  {
                                                      char *str = append1($1.str,'*',$3.str);
                                                      $$.str = strdup(str);
                                                      free(str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_terms '/' scalar_factors
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                      $$.str = strdup("length_mismatch_error");
                                                  else
                                                  {
                                                      char *str = append1($1.str,'/',$3.str);
                                                      $$.str = strdup(str);
                                                      free(str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_terms '^' scalar_factor    {char *str = append4("er.pows", '(', $1.str, ',', $3, ')'); $$.str = strdup(str); free(str); $$.size = $1.size;}
            ;


scalar_factors: EUCLIDEAN_LENGTHS '(' vector_exprs ')'           {char *str = append3("er.euclideanLengths", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | LENGTHS '(' edges ')'                            {char *str = append3("er.lengths", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | AREAS '(' faces ')'                              {char *str = append3("er.areas", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | VOLUMES '(' cells ')'                            {char *str = append3("er.volumes", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | DOTS '(' vector_exprs ',' vector_exprs ')'
                                                                 {
                                                                    if($3.size != $5.size)    // check that the lengths of the 2 terms are equal
                                                                        $$.str = strdup("length_mismatch_error");
                                                                    else
                                                                    {
                                                                        char *str = append4("er.dots", '(', $3.str, ',', $5.str, ')');
                                                                        $$.str = strdup(str);
                                                                        free(str);
                                                                        $$.size = $3.size;
                                                                    }
                                                                 }

              | CEILS '(' scalar_exprs ')'                       {char *str = append3("er.ceils", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | FLOORS '(' scalar_exprs ')'                      {char *str = append3("er.floors", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | ABSES '(' scalar_exprs ')'                       {char *str = append3("er.abses", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | VARIABLE                                         {
                                                                    if(strcmp(getType($1), "scalars") != 0)
                                                                    {
                                                                        $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                                        $$.str = strcat($$.str, "  ");
                                                                        $$.str = strcat($$.str, $1);
                                                                    }
                                                                    else
                                                                    {
                                                                        $$.str = strdup($1);
                                                                        $$.size = getSize1($1);
                                                                    }
                                                                 }
              ;


numbers: NUMBER                         {$$ = strdup($1);}
       | numbers ',' NUMBER             {char *str = append5($1,',',$3); $$ = strdup(str); free(str);}
       ;


vector_expr: vector_term                      {$$ = strdup($1);}
           | '-' vector_term                  {char *str = append7('-',$2); $$ = strdup(str); free(str);}
           | vector_expr '+' vector_term      {char *str = append1($1,'+',$3); $$ = strdup(str); free(str);}
           | vector_expr '-' vector_term      {char *str = append1($1,'-',$3); $$ = strdup(str); free(str);}
           ;


vector_term: '(' numbers ')'                       {char *str = append2('(', $2, ')'); $$ = strdup(str); free(str);}
           | CENTROID '(' cell ')'                 {char *str = append3("er.centroid", '(', $3, ')'); $$ = strdup(str); free(str);}
           | NORMAL '(' face ')'                   {char *str = append3("er.normal", '(', $3, ')'); $$ = strdup(str); free(str);}
           | '(' vector_expr ')'                   {char *str = append2('(', $2, ')'); $$ = strdup(str); free(str);}          // produces 1 shift/reduce conflict
           | vector_term '*' scalar_factor         {char *str = append1($1,'*',$3); $$ = strdup(str); free(str);}             // produces 1 reduce/reduce conflict
           | scalar_factor '*' vector_term         {char *str = append1($1,'*',$3); $$ = strdup(str); free(str);}
           | vector_term '/' scalar_factor         {char *str = append1($1,'/',$3); $$ = strdup(str); free(str);}
           | VARIABLE                              {
                                                      if(strcmp(getType($1), "vector") != 0)
                                                      {
                                                          $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                          $$ = strcat($$, "  ");
                                                          $$ = strcat($$, $1);
                                                      }
                                                      else
                                                      {
                                                          $$ = strdup($1);
                                                      }
                                                   }
           ;


vectors: vector_term                      {$$.str = strdup($1); $$.size = 1;}
       | vectors ',' vector_term          {char *str = append5($1.str,',',$3); $$.str = strdup(str); free(str); $$.size = $1.size + 1;}
       ;


vector_exprs: vector_terms                       {$$.str = strdup($1.str); $$.size = $1.size;}
            | '-' vector_terms                   {char *str = append7('-',$2.str); $$.str = strdup(str); free(str); $$.size = $2.size;}            // produces 1 shift/reduce conflict
            | vector_exprs '+' vector_terms
                                                 {
                                                    if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                        $$.str = strdup("length_mismatch_error");
                                                    else
                                                    {
                                                        char *str = append1($1.str,'+',$3.str);
                                                        $$.str = strdup(str);
                                                        free(str);
                                                        $$.size = $1.size;
                                                    }
                                                 }

            | vector_exprs '-' vector_terms
                                                 {
                                                    if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                        $$.str = strdup("length_mismatch_error");
                                                    else
                                                    {
                                                        char *str = append1($1.str,'-',$3.str);
                                                        $$.str = strdup(str);
                                                        free(str);
                                                        $$.size = $1.size;
                                                    }
                                                 }
            ;


vector_terms: '(' vectors ')'                        {char *str = append2('(', $2.str, ')'); $$.str = strdup(str); free(str); $$.size = $2.size;}
            | CENTROIDS '(' cells ')'                {char *str = append3("er.centroids", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
            | NORMALS '(' faces ')'                  {char *str = append3("er.normals", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
            | '(' vector_exprs ')'                   {char *str = append2('(', $2.str, ')'); $$.str = strdup(str); free(str); $$.size = $2.size;}          // produces 1 shift/reduce conflict
            | vector_terms '*' scalar_factor         {char *str = append1($1.str,'*',$3); $$.str = strdup(str); free(str); $$.size = $1.size;}             // produces 1 reduce/reduce conflict
            | scalar_factor '*' vector_terms         {char *str = append1($1,'*',$3.str); $$.str = strdup(str); free(str); $$.size = $3.size;}
            | vector_terms '/' scalar_factor         {char *str = append1($1.str,'/',$3); $$.str = strdup(str); free(str); $$.size = $1.size;}
            | VARIABLE                               {
                                                        if(strcmp(getType($1), "vectors") != 0)
                                                        {
                                                            $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                            $$.str = strcat($$.str, "  ");
                                                            $$.str = strcat($$.str, $1);
                                                        }
                                                        else
                                                        {
                                                            $$.str = strdup($1);
                                                            $$.size = getSize1($1);
                                                        }
                                                     }
            ;


vertex: VARIABLE           {
                              if(strcmp(getType($1), "vertex") != 0)
                              {
                                  $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                  $$ = strcat($$, "  ");
                                  $$ = strcat($$, $1);
                              }
                              else
                              {
                                  $$ = strdup($1);
                              }
                           }
      ;


vertices: INTERIOR_VERTICES '(' GRID ')'      {$$.str = strdup("er.interiorVertices()"); $$.size = INTERIORVERTICES;}
        | BOUNDARY_VERTICES '(' GRID ')'      {$$.str = strdup("er.boundaryVertices()"); $$.size = BOUNDARYVERTICES;}
        | ALL_VERTICES '(' GRID ')'           {$$.str = strdup("er.allVertices()"); $$.size = ALLVERTICES;}
        | VARIABLE                            {
                                                  if(strcmp(getType($1), "vertices") != 0)
                                                  {
                                                      $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                      $$.str = strcat($$.str, "  ");
                                                      $$.str = strcat($$.str, $1);
                                                  }
                                                  else
                                                  {
                                                      $$.str = strdup($1);
                                                      $$.size = getSize1($1);
                                                  }
                                              }
        ;


edge: VARIABLE             {
                              if(strcmp(getType($1), "edge") != 0)
                              {
                                  $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                  $$ = strcat($$, "  ");
                                  $$ = strcat($$, $1);
                              }
                              else
                              {
                                  $$ = strdup($1);
                              }
                           }
    ;


edges: INTERIOR_EDGES '(' GRID ')'      {$$.str = strdup("er.interiorEdges()"); $$.size = INTERIOREDGES;}
     | BOUNDARY_EDGES '(' GRID ')'      {$$.str = strdup("er.boundaryEdges()"); $$.size = BOUNDARYEDGES;}
     | ALL_EDGES '(' GRID ')'           {$$.str = strdup("er.allEdges()"); $$.size = ALLEDGES;}
     | VARIABLE                         {
                                            if(strcmp(getType($1), "edges") != 0)
                                            {
                                                $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                $$.str = strcat($$.str, "  ");
                                                $$.str = strcat($$.str, $1);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1);
                                                $$.size = getSize1($1);
                                            }
                                        }
     ;


face: VARIABLE                    {
                                      if(strcmp(getType($1), "face") != 0)
                                      {
                                          $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                          $$ = strcat($$, "  ");
                                          $$ = strcat($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }
    ;


faces: INTERIOR_FACES '(' GRID ')'      {$$.str = strdup("er.interiorFaces()"); $$.size = INTERIORFACES;}
     | BOUNDARY_FACES '(' GRID ')'      {$$.str = strdup("er.boundaryFaces()"); $$.size = BOUNDARYFACES;}
     | ALL_FACES '(' GRID ')'           {$$.str = strdup("er.allFaces()"); $$.size = ALLFACES;}
     | VARIABLE                         {
                                            if(strcmp(getType($1), "faces") != 0)
                                            {
                                                $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                $$.str = strcat($$.str, "  ");
                                                $$.str = strcat($$.str, $1);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1);
                                                $$.size = getSize1($1);
                                            }
                                        }
     ;


cell: FIRST_CELL '(' face ')'     {char *str = append3("er.firstCell", '(', $3, ')'); $$ = strdup(str); free(str);}
    | SECOND_CELL '(' face ')'    {char *str = append3("er.secondCell", '(', $3, ')'); $$ = strdup(str); free(str);}
    | VARIABLE                    {
                                      if(strcmp(getType($1), "cell") != 0)
                                      {
                                          $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                          $$ = strcat($$, "  ");
                                          $$ = strcat($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }
    ;


cells: INTERIOR_CELLS '(' GRID ')'          {$$.str = strdup("er.interiorCells()"); $$.size = INTERIORCELLS;}
     | BOUNDARY_CELLS '(' GRID ')'          {$$.str = strdup("er.boundaryCells()"); $$.size = BOUNDARYCELLS;}
     | ALL_CELLS '(' GRID ')'               {$$.str = strdup("er.allCells()"); $$.size = ALLCELLS;}
     | FIRST_CELLS '(' faces ')'            {char *str = append3("er.firstCells", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
     | SECOND_CELLS '(' faces ')'           {char *str = append3("er.secondCells", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
     | VARIABLE                             {
                                                if(strcmp(getType($1), "cells") != 0)
                                                {
                                                    $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                    $$.str = strcat($$.str, "  ");
                                                    $$.str = strcat($$.str, $1);
                                                }
                                                else
                                                {
                                                    $$.str = strdup($1);
                                                    $$.size = getSize1($1);
                                                }
                                            }
     ;


adb: GRADIENT '(' adb ')'         {char *str = append3("er.negGradient", '(', $3, ')'); $$ = strdup(str); free(str);}
   | DIVERGENCE '(' adb ')'       {char *str = append3("er.divergence", '(', $3, ')'); $$ = strdup(str); free(str);}
   | VARIABLE                     {
                                      if(strcmp(getType($1), "scalarAD") != 0)
                                      {
                                          $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                          $$ = strcat($$, "  ");
                                          $$ = strcat($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }
   ;


adbs: GRADIENTS '(' adbs ')'      {char *str = append3("er.negGradients", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
    | DIVERGENCES '(' adbs ')'    {char *str = append3("er.divergences", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
    | VARIABLE                    {
                                      if(strcmp(getType($1), "scalarsAD") != 0)
                                      {
                                          $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                          $$.str = strcat($$.str, "  ");
                                          $$.str = strcat($$.str, $1);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1);
                                          $$.size = getSize1($1);
                                      }
                                  }
    ;


boolean_expr: boolean_term                           {$$ = strdup($1);}
            | NOT boolean_term                       {char *str = append7('!', $2); $$ = strdup(str); free(str);}
            | boolean_expr AND boolean_term          {char *str = append6($1, "&&", $3); $$ = strdup(str); free(str);}
            | boolean_expr OR boolean_term           {char *str = append6($1, "||", $3); $$ = strdup(str); free(str);}
            | boolean_expr XOR boolean_term          {char *str = append10($1, $3); $$ = strdup(str); free(str);}
            ;



boolean_term: TRUE                                   {$$ = strdup("true");}
            | FALSE                                  {$$ = strdup("false");}
            | scalar_expr '>' scalar_expr            {char *str = append1($1, '>', $3); $$ = strdup(str); free(str);}
            | scalar_expr '<' scalar_expr            {char *str = append1($1, '<', $3); $$ = strdup(str); free(str);}
            | scalar_expr LESSEQ scalar_expr         {char *str = append6($1, "<=", $3); $$ = strdup(str); free(str);}
            | scalar_expr GREATEREQ scalar_expr      {char *str = append6($1, ">=", $3); $$ = strdup(str); free(str);}
            | scalar_expr EQ scalar_expr             {char *str = append6($1, "==", $3); $$ = strdup(str); free(str);}
            | scalar_expr NOTEQ scalar_expr          {char *str = append6($1, "!=", $3); $$ = strdup(str); free(str);}
            | '(' boolean_expr ')'                   {char *str = append2('(', $2, ')'); $$ = strdup(str); free(str);}
            | VARIABLE                               {
                                                        if(strcmp(getType($1), "bool") != 0)
                                                        {
                                                            $$ = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                            $$ = strcat($$, "  ");
                                                            $$ = strcat($$, $1);
                                                        }
                                                        else
                                                        {
                                                            $$ = strdup($1);
                                                        }
                                                    }
            ;


boolean_exprs: boolean_terms                           {$$.str = strdup($1.str); $$.size = $1.size;}
             | NOT boolean_terms                       {char *str = append7('!', $2.str); $$.str = strdup(str); free(str); $$.size = $2.size;}
             | boolean_exprs AND boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append6($1.str, "&&", $3.str);
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $1.size;
                                                          }
                                                       }

             | boolean_exprs OR boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append6($1.str, "||", $3.str);
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $1.size;
                                                          }
                                                       }

             | boolean_exprs XOR boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append10($1.str, $3.str);
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $1.size;
                                                          }
                                                       }
             ;



boolean_terms: '(' scalars ')' '>' '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append14('(',$2.str,')','>','(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' '<' '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append14('(',$2.str,')','<','(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' LESSEQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append15('(',$2.str,')',"<=",'(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' GREATEREQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append15('(',$2.str,')',">=",'(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' EQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append15('(',$2.str,')',"==",'(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' NOTEQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                              $$.str = strdup("length_mismatch_error");
                                                          else
                                                          {
                                                              char *str = append15('(',$2.str,')',"!=",'(',$6.str,')');
                                                              $$.str = strdup(str);
                                                              free(str);
                                                              $$.size = $2.size;
                                                          }
                                                      }
             | '(' boolean_exprs ')'                  {char *str = append2('(', $2.str, ')'); $$.str = strdup(str); free(str); $$.size = $2.size;}
             | VARIABLE                               {
                                                          if(strcmp(getType($1), "bools") != 0)
                                                          {
                                                              $$.str = strdup("wrong_type_error");   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                              $$.str = strcat($$.str, "  ");
                                                              $$.str = strcat($$.str, $1);
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup($1);
                                                              $$.size = getSize1($1);
                                                          }
                                                      }
             ;



plural: scalar_exprs            {$$.str = strdup($1.str); $$.size = $1.size;}
      | vector_exprs            {$$.str = strdup($1.str); $$.size = $1.size;}
      | vertices                {$$.str = strdup($1.str); $$.size = $1.size;}
      | edges                   {$$.str = strdup($1.str); $$.size = $1.size;}
      | faces                   {$$.str = strdup($1.str); $$.size = $1.size;}
      | cells                   {$$.str = strdup($1.str); $$.size = $1.size;}
      | adbs                    {$$.str = strdup($1.str); $$.size = $1.size;}
      | boolean_exprs           {$$.str = strdup($1.str); $$.size = $1.size;}
      ;

/*
header: SCALAR_VARIABLE HEADER_DECL SCALAR                          {char *str = append9("Scalar", ' ', $1); $$ = strdup(str); free(str);}
      | VECTOR_VARIABLE HEADER_DECL VECTOR                          {char *str = append9("Vector", ' ', $1); $$ = strdup(str); free(str);}
      | VERTEX_VARIABLE HEADER_DECL VERTEX                          {char *str = append9("Vertex", ' ', $1); $$ = strdup(str); free(str);}
      | EDGE_VARIABLE HEADER_DECL EDGE                              {char *str = append9("Edge", ' ', $1); $$ = strdup(str); free(str);}
      | FACE_VARIABLE HEADER_DECL FACE                              {char *str = append9("Face", ' ', $1); $$ = strdup(str); free(str);}
      | CELL_VARIABLE HEADER_DECL CELL                              {char *str = append9("Cell", ' ', $1); $$ = strdup(str); free(str);}
      | ADB_VARIABLE HEADER_DECL ADB                                {char *str = append9("ScalarAD", ' ', $1); $$ = strdup(str); free(str);}
      | BOOL_VARIABLE HEADER_DECL BOOLEAN                           {char *str = append9("bool", ' ', $1); $$ = strdup(str); free(str);}
      | SCALARS_VARIABLE HEADER_DECL COLLECTION OF SCALAR           {char *str = append9("CollOfScalars", ' ', $1); $$ = strdup(str); free(str);}
      | VECTORS_VARIABLE HEADER_DECL COLLECTION OF VECTOR           {char *str = append9("CollOfVectors", ' ', $1); $$ = strdup(str); free(str);}
      | VERTICES_VARIABLE HEADER_DECL COLLECTION OF VERTEX          {char *str = append9("CollOfVertices", ' ', $1); $$ = strdup(str); free(str);}
      | EDGES_VARIABLE HEADER_DECL COLLECTION OF EDGE               {char *str = append9("CollOfEdges", ' ', $1); $$ = strdup(str); free(str);}
      | FACES_VARIABLE HEADER_DECL COLLECTION OF EDGE               {char *str = append9("CollOfFaces", ' ', $1); $$ = strdup(str); free(str);}
      | CELLS_VARIABLE HEADER_DECL COLLECTION OF FACE               {char *str = append9("CollOfCells", ' ', $1); $$ = strdup(str); free(str);}
      | ADBS_VARIABLE HEADER_DECL COLLECTION OF ADB                 {char *str = append9("CollOfScalarsAD", ' ', $1); $$ = strdup(str); free(str);}
      | BOOLS_VARIABLE HEADER_DECL COLLECTION OF BOOLEAN            {char *str = append9("CollOfBools", ' ', $1); $$ = strdup(str); free(str);}
      ;


parameter_list: header                         {$$ = strdup($1);}
              | parameter_list ',' header      {char *str = append5($1,',',$3); $$ = strdup(str); free(str);}
              ;


commands: command                              {$$ = strdup($1);}
        | commands end_lines command           {char *str = append6($1, $2, $3); $$ = strdup(str); free(str);}
        ;


type: SCALAR                                {$$.str = strdup("Scalar"); $$.size = ANY;}
    | VECTOR                                {$$.str = strdup("Vector"); $$.size = ANY;}
    | VERTEX                                {$$.str = strdup("Vertex"); $$.size = ANY;}
    | EDGE                                  {$$.str = strdup("Edge"); $$.size = ANY;}
    | FACE                                  {$$.str = strdup("Face"); $$.size = ANY;}
    | CELL                                  {$$.str = strdup("Cell"); $$.size = ANY;}
    | ADB                                   {$$.str = strdup("ScalarAD"); $$.size = ANY;}
    | BOOLEAN                               {$$.str = strdup("bool"); $$.size = ANY;}
    | COLLECTION OF SCALAR                  {$$.str = strdup("CollOfScalars"); $$.size = ANY;}
    | COLLECTION OF VECTOR                  {$$.str = strdup("CollOfVectors"); $$.size = ANY;}
    | COLLECTION OF VERTEX                  {$$.str = strdup("CollOfVertices"); $$.size = ANY;}
    | COLLECTION OF EDGE                    {$$.str = strdup("CollOfEdges"); $$.size = ANY;}
    | COLLECTION OF FACE                    {$$.str = strdup("CollOfFaces"); $$.size = ANY;}
    | COLLECTION OF CELL                    {$$.str = strdup("CollOfCells"); $$.size = ANY;}
    | COLLECTION OF ADB                     {$$.str = strdup("CollOfScalarsAD"); $$.size = ANY;}
    | COLLECTION OF BOOLEAN                 {$$.str = strdup("CollOfBools"); $$.size = ANY;}
    | COLLECTION OF SCALAR ON plural        {$$.str = strdup("CollOfScalars"); $$.size = $5.size;}
    | COLLECTION OF VECTOR ON plural        {$$.str = strdup("CollOfVectors"); $$.size = $5.size;}
    | COLLECTION OF VERTEX ON plural        {$$.str = strdup("CollOfVertices"); $$.size = $5.size;}
    | COLLECTION OF EDGE ON plural          {$$.str = strdup("CollOfEdges"); $$.size = $5.size;}
    | COLLECTION OF FACE ON plural          {$$.str = strdup("CollOfFaces"); $$.size = $5.size;}
    | COLLECTION OF CELL ON plural          {$$.str = strdup("CollOfCells"); $$.size = $5.size;}
    | COLLECTION OF ADB ON plural           {$$.str = strdup("CollOfScalarsAD"); $$.size = $5.size;}
    | COLLECTION OF BOOLEAN ON plural       {$$.str = strdup("CollOfBools"); $$.size = $5.size;}
    ;

*/

/*
value: scalar            {$$.str = strdup($1); $$.size = ANY;}
     | vector            {$$.str = strdup($1); $$.size = ANY;}
     | vertex            {$$.str = strdup($1); $$.size = ANY;}
     | edge              {$$.str = strdup($1); $$.size = ANY;}
     | face              {$$.str = strdup($1); $$.size = ANY;}
     | cell              {$$.str = strdup($1); $$.size = ANY;}
     | adb               {$$.str = strdup($1); $$.size = ANY;}
     | boolean           {$$.str = strdup($1); $$.size = ANY;}
     | scalar_exprs      {$$.str = strdup($1.size); $$.size = $1.size;}
     | vector_exprs      {$$.str = strdup($1.size); $$.size = $1.size;}
     | vertices          {$$.str = strdup($1.size); $$.size = $1.size;}
     | edges             {$$.str = strdup($1.size); $$.size = $1.size;}
     | faces             {$$.str = strdup($1.size); $$.size = $1.size;}
     | cells             {$$.str = strdup($1.size); $$.size = $1.size;}
     | adbs              {$$.str = strdup($1.size); $$.size = $1.size;}
     | booleans          {$$.str = strdup($1.size); $$.size = $1.size;}
     ;


values: value                   {$$ = strdup($1);}
      | values ',' value        {char *str = append5($1,',',$3); $$ = strdup(str); free(str);}
      ;
*/



/*
end_lines: '\n'                 {char *s = (char *)malloc(sizeof(char)); s[0] = '\n'; s[1] = '\0'; $$ = strdup(s); free(s);}
         | '\n' end_lines       {char *str = append7('\n', $2); $$ = strdup(str); free(str);}
         |                      {}
         ;


return_instr: RETURN boolean '?' VARIABLE ':' VARIABLE
                  {
                    if(check5($4.str) == false || check5($6.str) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.size = -1;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        sprintf($$.str, "return %s ? %s : %s", $2, $4.str, $6.str);
                        $$.size = getSize3($4.str);
                    }
                  }

            | RETURN VARIABLE
                  {
                    if(check5($2.str) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.size = -1;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        sprintf($$.str, "return %s", $2.str);
                        $$.size = getSize3($2.str);
                    }
                  }
            ;


function_start: FUNCTION_VARIABLE '=' end_lines '{'
                                            {
                                              insideFunction = true;
                                              currentFunctionIndex = getIndex2($1);
                                              char *str = append12($1, '=', $3, '{');
                                              $$ = strdup(str);
                                              free(str);
                                            }


// these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type
                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name, $1) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                    sprintf($$, "error: The function '%s' is redeclared", $1);
                                                else
                                                {
                                                        fun[funNo++].name = strdup($1);
                                                        fun[funNo-1].returnType = strdup($8.str);
                                                        fun[funNo-1].returnSize = $8.size;
                                                        fun[funNo-1].paramList = strdup($5);
                                                        fun[funNo-1].noParam = find4($5);
                                                        fun[funNo-1].assigned = false;
                                                        char *str = append11($8.str, $1, '(', $5, ')');
                                                        $$ = strdup(str);
                                                        free(str);
                                                }
                                            }
                    ;

function_assignment: function_start end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name, extract($1)) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                      if(fun[i].assigned == true)
                                                          sprintf($$, "error: The function '%s' is reassigned", fun[i].name);
                                                      else
                                                      {
                                                            if(getSize2(fun[i].name) != $5.size)
                                                                if(getSize2(fun[i].name) == ANY && $5.size != -1)
                                                                {
                                                                    char *str = append13($1, $2, $3, $4, $5.str, $6, '}');
                                                                    $$ = strdup(str);
                                                                    free(str);
                                                                    fun[i].returnSize = $5.size;
                                                                    fun[i].assigned = true;
                                                                }
                                                                else
                                                                    sprintf($$, "error: The return type of the function '%s' from its assignment differs than the length of the return type from the function's definition", fun[i].name);
                                                            else
                                                            {
                                                                char *str = append13($1, $2, $3, $4, $5.str, $6, '}');
                                                                $$ = strdup(str);
                                                                free(str);
                                                                fun[i].assigned = true;
                                                            }
                                                      }
                                                else
                                                {
                                                    sprintf($$, "error: The function '%s' must be declared before being assigned", extract($1));
                                                }
                                                insideFunction = false;
                                                currentFunctionIndex = -1;
                                            }
                   ;




// function_declaration_with_assignment: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type '=' end_lines '{' end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
//                                     ; // tre sa punem booleana globala true inainte sa execute comenzile din functie

// function_call: FUNCTION_VARIABLE '(' values ')'     // can be an instruction in the body of another function
//              ;   // it must be put at every type where there exists a variable(s) (scalar_expr etc)


*/




singular_declaration: VARIABLE ':' SCALAR               {$$ = singular_declaration_function($1, "scalar");}
                    | VARIABLE ':' VECTOR               {$$ = singular_declaration_function($1, "vector");}
                    | VARIABLE ':' VERTEX               {$$ = singular_declaration_function($1, "vertex");}
                    | VARIABLE ':' EDGE                 {$$ = singular_declaration_function($1, "edge");}
                    | VARIABLE ':' FACE                 {$$ = singular_declaration_function($1, "face");}
                    | VARIABLE ':' CELL                 {$$ = singular_declaration_function($1, "cell");}
                    | VARIABLE ':' ADB                  {$$ = singular_declaration_function($1, "scalarAD");}
                    | VARIABLE ':' BOOLEAN              {$$ = singular_declaration_function($1, "bool");}
                    ;


plural_declaration: VARIABLE ':' COLLECTION OF SCALAR       {$$ = plural_declaration_function($1, "scalars");}
                  | VARIABLE ':' COLLECTION OF VECTOR       {$$ = plural_declaration_function($1, "vectors");}
                  | VARIABLE ':' COLLECTION OF VERTEX       {$$ = plural_declaration_function($1, "vertices");}
                  | VARIABLE ':' COLLECTION OF EDGE         {$$ = plural_declaration_function($1, "edges");}
                  | VARIABLE ':' COLLECTION OF FACE         {$$ = plural_declaration_function($1, "faces");}
                  | VARIABLE ':' COLLECTION OF CELL         {$$ = plural_declaration_function($1, "cells");}
                  | VARIABLE ':' COLLECTION OF ADB          {$$ = plural_declaration_function($1, "scalarsAD");}
                  | VARIABLE ':' COLLECTION OF BOOLEAN      {$$ = plural_declaration_function($1, "bools");}
                  ;


extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON plural      {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "scalars", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF VECTOR ON plural      {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "vectors", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF VERTEX ON plural      {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "vertices", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF EDGE ON plural        {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "edges", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF FACE ON plural        {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "faces", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF CELL ON plural        {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "cells", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF ADB ON plural         {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "scalarsAD", st, $7.size);}
                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural     {char *st = structureToString($7.str); $$ = extended_plural_declaration_function($1, "bools", st, $7.size);}
                           ;


declaration: singular_declaration           {$$ = strdup($1);}
           | plural_declaration             {$$ = strdup($1);}
           | extended_plural_declaration    {$$ = strdup($1);}
           ;



singular_assignment: VARIABLE '=' scalar_expr              {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "scalar", str, "Scalar");}
                   | VARIABLE '=' vector_expr              {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "vector", str, "Vector");}
                   | VARIABLE '=' vertex                   {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "vertex", str, "Vertex");}
                   | VARIABLE '=' edge                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "edge", str, "Edge");}
                   | VARIABLE '=' face                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "face", str, "Face");}
                   | VARIABLE '=' cell                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "cell", str, "Cell");}
                   | VARIABLE '=' adb                      {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "scalarAD", str, "ScalarAD");}
                   | VARIABLE '=' boolean_expr             {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; $$ = singular_assignment_function($1, "bool", str, "bool");}
                   ;


plural_assignment: VARIABLE '=' scalar_exprs              {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "scalars", st, "CollOfScalars", $3.size);}
                 | VARIABLE '=' vector_exprs              {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "vectors", st, "CollOfVectors", $3.size);}
                 | VARIABLE '=' vertices                  {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "vertices", st, "CollOfVertices", $3.size);}
                 | VARIABLE '=' edges                     {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "edges", st, "CollOfEdges", $3.size);}
                 | VARIABLE '=' faces                     {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "faces", st, "CollOfFaces", $3.size);}
                 | VARIABLE '=' cells                     {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "cells", st, "CollOfCells", $3.size);}
                 | VARIABLE '=' adbs                      {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $3.size);}
                 | VARIABLE '=' boolean_exprs             {char *st = structureToString($3.str); $$ = plural_assignment_function($1, "bools", st, "CollOfBools", $3.size);}
                 ;


//if the variable hasn't been declared before, it is an assignment with deduced declaration (type)

assignment: singular_assignment     {$$ = strdup($1);}
          | plural_assignment       {$$ = strdup($1);}
          ;




singular_declaration_with_assignment: VARIABLE ':' SCALAR '=' scalar_expr          {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "scalar", str, "Scalar");}
                                    | VARIABLE ':' VECTOR '=' vector_expr          {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "vector", str, "Vector");}
                                    | VARIABLE ':' VERTEX '=' vertex               {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "vertex", str, "Vertex");}
                                    | VARIABLE ':' EDGE '=' edge                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "edge", str, "Edge");}
                                    | VARIABLE ':' FACE '=' face                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "face", str, "Face");}
                                    | VARIABLE ':' CELL '=' cell                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "cell", str, "Cell");}
                                    | VARIABLE ':' ADB '=' adb                     {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "scalarAD", str, "ScalarAD");}
                                    | VARIABLE ':' BOOLEAN '=' boolean_expr        {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; $$ = singular_declaration_with_assignment_function($1, "bool", str, "bool");}
                                    ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' scalar_exprs        {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "scalars", st, "CollOfScalars", $7.size);}
                                  | VARIABLE ':' COLLECTION OF VECTOR '=' vector_exprs        {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "vectors", st, "CollOfVectors", $7.size);}
                                  | VARIABLE ':' COLLECTION OF VERTEX '=' vertices            {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "vertices", st, "CollOfVertices", $7.size);}
                                  | VARIABLE ':' COLLECTION OF EDGE '=' edges                 {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "edges", st, "CollOfEdges", $7.size);}
                                  | VARIABLE ':' COLLECTION OF FACE '=' faces                 {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "faces", st, "CollOfFaces", $7.size);}
                                  | VARIABLE ':' COLLECTION OF CELL '=' cells                 {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "cells", st, "CollOfCells", $7.size);}
                                  | VARIABLE ':' COLLECTION OF ADB '=' adbs                   {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $7.size);}
                                  | VARIABLE ':' COLLECTION OF BOOLEAN '=' boolean_exprs      {char *st = structureToString($7.str); $$ = plural_declaration_with_assignment_function($1, "bools", st, "CollOfBools", $7.size);}
                                  ;


extended_plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR ON plural '=' scalar_exprs        {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "scalars", st, "CollOfScalars", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF VECTOR ON plural '=' vector_exprs        {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "vectors", st, "CollOfVectors", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF VERTEX ON plural '=' vertices            {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "vertices", st, "CollOfVertices", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF EDGE ON plural '=' edges                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "edges", st, "CollOfEdges", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF FACE ON plural '=' faces                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "faces", st, "CollOfFaces", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF CELL ON plural '=' cells                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "cells", st, "CollOfCells", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF ADB ON plural '=' adbs                   {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", st2, $9.size, $7.size);}
                                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural '=' boolean_exprs      {char *st = structureToString($9.str); char *st2 = structureToString($7.str); $$ = extended_plural_declaration_with_assignment_function($1, "bools", st, "CollOfBools", st2, $9.size, $7.size);}
                                           ;


 declaration_with_assignment: singular_declaration_with_assignment          {$$ = strdup($1);}
                            | plural_declaration_with_assignment            {$$ = strdup($1);}
                            | extended_plural_declaration_with_assignment   {$$ = strdup($1);}
                            ;




// instructions which can be used in the program and in a function's body
command: declaration                    {$$ = strdup($1);}
       | assignment                     {$$ = strdup($1);}
       | declaration_with_assignment    {$$ = strdup($1);}
       ;


// instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                    {$$ = strdup($1);}
    //  | function_declaration                       {$$ = strdup($1);}
    //  | function_assignment                        {$$ = strdup($1);}
    //  | function_declaration_with_assignment       {$$ = strdup($1);}
        ;


pr: pr command2 '\n'                  {printf("%s", $2); printf("\n"); currentLineNumber++;}
  | pr command2 COMMENT '\n'          {printf("%s", $2); printf(" //"); int i; for(i = 1; i < strlen($3); i++) printf("%c", $3[i]); printf("\n"); currentLineNumber++;}
  | pr COMMENT '\n'                   {printf("//"); int i; for(i = 1; i < strlen($2); i++) printf("%c", $2[i]); printf("\n"); currentLineNumber++;}
  | pr '\n'                           {printf("\n"); currentLineNumber++;}
  |                                   {}
  ;

%%


extern int yylex();
extern int yyparse();
int main()
{
printf("Opm::parameter::ParameterGroup param(argc, argv, false);\nEquelleRuntimeCPU er(param);\nUserParameters up(param, er);\n\n");
yyparse();
return 0;
}


void yyerror(const char* s)
{
printf("%s",s);
}


char* append1(char *s1, char s2, char *s3)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+3));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = ' ';
  str[strlen(s1)+1] = s2;
  str[strlen(s1)+2] = ' ';
  for(i = 0; i < strlen(s3); i++)
    str[3+strlen(s1)+i] = s3[i];
  str[strlen(s1)+strlen(s3)+3] = '\0';
  return str;
}


char* append2(char s1, char *s2, char s3)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s2)+2));
  str[0] = s1;
  int i;
  for(i = 0; i < strlen(s2); i++)
    str[i+1] = s2[i];
  str[strlen(s2)+1] = s3;
  str[strlen(s2)+2] = '\0';
  return str;
}


char* append3(char *s1, char s2, char *s3, char s4)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+2));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = s2;
  for(i = 0; i < strlen(s3); i++)
    str[1+strlen(s1)+i] = s3[i];
  str[strlen(s1)+strlen(s3)+1] = s4;
  str[strlen(s1)+strlen(s3)+2] = '\0';
  return str;
}


char* append4(char *s1, char s2, char *s3, char s4, char *s5, char s6)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+strlen(s5)+4));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = s2;
  for(i = 0; i < strlen(s3); i++)
    str[1+strlen(s1)+i] = s3[i];
  str[strlen(s1)+strlen(s3)+1] = s4;
  str[strlen(s1)+strlen(s3)+2] = ' ';
  for(i = 0; i < strlen(s5); i++)
    str[strlen(s1)+strlen(s3)+3+i] = s5[i];
  str[strlen(s1)+strlen(s3)+strlen(s5)+3] = s6;
  str[strlen(s1)+strlen(s3)+strlen(s5)+4] = '\0';
  return str;
}


char* append5(char *s1, char s2, char *s3)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+2));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = s2;
  str[strlen(s1)+1] = ' ';
  for(i = 0; i < strlen(s3); i++)
    str[2+strlen(s1)+i] = s3[i];
  str[strlen(s1)+strlen(s3)+2] = '\0';
  return str;
}


char *append6(char *s1, char *s2, char *s3)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s2)+strlen(s3)+2));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = ' ';
  for(i = 0; i < strlen(s2); i++)
    str[1+strlen(s1)+i] = s2[i];
  str[strlen(s1)+strlen(s2)+1] = ' ';
  for(i = 0; i < strlen(s3); i++)
    str[2+strlen(s1)+strlen(s2)+i] = s3[i];
  str[strlen(s1)+strlen(s2)+strlen(s3)+2] = '\0';
  return str;
}


char* append7(char s1, char *s2)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s2)+1));
  str[0] = s1;
  int i;
  for(i = 0; i < strlen(s2); i++)
    str[i+1] = s2[i];
  str[strlen(s2)+1] = '\0';
  return str;
}


char* append8(char s1, char s2, char *s3, char s4)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s3)+3));
  str[0] = s1;
  str[1] = s2;
  int i;
  for(i = 0; i < strlen(s3); i++)
    str[i+2] = s3[i];
  str[strlen(s3)+2] = s4;
  str[strlen(s3)+3] = '\0';
  return str;
}


char* append9(char *s1, char s2, char *s3)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+1));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = s2;
  for(i = 0; i < strlen(s3); i++)
    str[1+strlen(s1)+i] = s3[i];
  str[strlen(s1)+strlen(s3)+1] = '\0';
  return str;
}


char* append10(char *s1, char *s2)   // function which returns the C++ code for the XOR between the two given variables: (s1 && (!s2)) || (s2 && (!s1))
{
  char *str = (char*)malloc(5*sizeof(char)*(2*(strlen(s1)+strlen(s2))+22));
  str[0] = '(';
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i+1] = s1[i];
  str[1+strlen(s1)] = ' ';
  str[2+strlen(s1)] = '&';
  str[3+strlen(s1)] = '&';
  str[4+strlen(s1)] = ' ';
  str[5+strlen(s1)] = '(';
  str[6+strlen(s1)] = '!';
  for(i = 0; i < strlen(s2); i++)
    str[strlen(s1)+i+7] = s2[i];
  str[7+strlen(s1)+strlen(s2)] = ')';
  str[8+strlen(s1)+strlen(s2)] = ')';
  str[9+strlen(s1)+strlen(s2)] = ' ';
  str[10+strlen(s1)+strlen(s2)] = '|';
  str[11+strlen(s1)+strlen(s2)] = '|';
  str[12+strlen(s1)+strlen(s2)] = ' ';
  str[13+strlen(s1)+strlen(s2)] = '(';
  for(i = 0; i < strlen(s2); i++)
    str[strlen(s1)+strlen(s2)+i+14] = s2[i];
  str[strlen(s1)+2*strlen(s2)+14] = ' ';
  str[strlen(s1)+2*strlen(s2)+15] = '&';
  str[strlen(s1)+2*strlen(s2)+16] = '&';
  str[strlen(s1)+2*strlen(s2)+17] = ' ';
  str[strlen(s1)+2*strlen(s2)+18] = '(';
  str[strlen(s1)+2*strlen(s2)+19] = '!';
  for(i = 0; i < strlen(s1); i++)
    str[strlen(s1)+2*strlen(s2)+i+20] = s1[i];
  str[2*(strlen(s1)+strlen(s2))+20] = ')';
  str[2*(strlen(s1)+strlen(s2))+21] = ')';
  str[2*(strlen(s1)+strlen(s2))+22] = '\0';
  return str;
}


char* append11(char *s1, char *s2, char s3, char *s4, char s5)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s2)+strlen(s4)+3));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = ' ';
  for(i = 0; i < strlen(s2); i++)
    str[strlen(s1)+1+i] = s2[i];
  str[strlen(s1)+strlen(s2)+1] = s3;
  for(i = 0; i < strlen(s4); i++)
    str[2+strlen(s1)+strlen(s2)+i] = s4[i];
  str[2+strlen(s1)+strlen(s2)+strlen(s4)+2] = s5;
  str[2+strlen(s1)+strlen(s2)+strlen(s4)+3] = '\0';
  return str;
}


char* append12(char *s1, char s2, char *s3, char s4)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s3)+4));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  str[strlen(s1)] = ' ';
  str[strlen(s1)+1] = s2;
  str[strlen(s1)+2] = ' ';
  for(i = 0; i < strlen(s3); i++)
    str[strlen(s1)+3+i] = s3[i];
  str[strlen(s1)+strlen(s3)+3] = s4;
  str[strlen(s1)+strlen(s3)+4] = '\0';
  return str;
}


char* append13(char *s1, char *s2, char *s3, char *s4, char *s5, char *s6, char s7)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s1)+strlen(s2)+strlen(s3)+strlen(s4)+strlen(s5)+strlen(s6)+1));
  int i;
  for(i = 0; i < strlen(s1); i++)
    str[i] = s1[i];
  for(i = 0; i < strlen(s2); i++)
    str[strlen(s1)+i] = s2[i];
  for(i = 0; i < strlen(s3); i++)
    str[strlen(s1)+strlen(s2)+i] = s3[i];
  for(i = 0; i < strlen(s4); i++)
    str[strlen(s1)+strlen(s2)+strlen(s3)+i] = s4[i];
  for(i = 0; i < strlen(s5); i++)
    str[strlen(s1)+strlen(s2)+strlen(s3)+strlen(s4)+i] = s5[i];
  for(i = 0; i < strlen(s6); i++)
    str[strlen(s1)+strlen(s2)+strlen(s3)+strlen(s4)+strlen(s5)+i] = s6[i];
  str[strlen(s1)+strlen(s2)+strlen(s3)+strlen(s4)+strlen(s5)+strlen(s6)] = s7;
  str[strlen(s1)+strlen(s2)+strlen(s3)+strlen(s4)+strlen(s5)+strlen(s6)+1] = '\0';
  return str;
}


char* append14(char s1, char *s2, char s3, char s4, char s5, char *s6, char s7)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s2)+strlen(s6)+7));
  int i;
  str[0] = s1;
  for(i = 0; i < strlen(s2); i++)
    str[1+i] = s2[i];
  str[1+strlen(s2)] = s3;
  str[2+strlen(s2)] = ' ';
  str[3+strlen(s2)] = s4;
  str[4+strlen(s2)] = ' ';
  str[5+strlen(s2)] = s5;
  for(i = 0; i < strlen(s6); i++)
    str[6+strlen(s2)+i] = s6[i];
  str[6+strlen(s2)+strlen(s6)] = s7;
  str[7+strlen(s2)+strlen(s6)] = '\0';
  return str;
}


char* append15(char s1, char *s2, char s3, char *s4, char s5, char *s6, char s7)
{
  char *str = (char*)malloc(5*sizeof(char)*(strlen(s2)+strlen(s4)+strlen(s6)+6));
  int i;
  str[0] = s1;
  for(i = 0; i < strlen(s2); i++)
    str[1+i] = s2[i];
  str[1+strlen(s2)] = s3;
  str[2+strlen(s2)] = ' ';
  for(i = 0; i < strlen(s4); i++)
    str[3+strlen(s2)+i] = s4[i];
  str[3+strlen(s2)+strlen(s4)] = ' ';
  str[4+strlen(s2)+strlen(s4)] = s5;
  for(i = 0; i < strlen(s6); i++)
    str[5+strlen(s2)+strlen(s4)+i] = s6[i];
  str[5+strlen(s2)+strlen(s4)+strlen(s6)] = s7;
  str[6+strlen(s2)+strlen(s4)+strlen(s6)] = '\0';
  return str;
}


bool find1(char *s1, char *s2)     // function which returns true if s2 is contained in s1
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(strcmp(pch, s2) == 0)
          return true;
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return false;
}


char* find2(char *s1)   // function which returns the first undeclared variable from a given expression (this function is called after the function "check1" returns false)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0)  // not a function
        {
            if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
            {
                bool found = false;
                int i;
                for(i = 0; i < varNo; i++)
                {
                    if(strcmp(pch, var[i].name) == 0)
                    {
                      found = true;
                      break;
                    }
                }
                if(found == false)
                  return pch;
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return strdup("InvalidCall");
}


char* find3(char *s1)     // function which returns the first unassigned variable from a given expression (this function is called after the function "check2" returns false)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < varNo; i++)
      {
          if(strcmp(pch, var[i].name) == 0)
          {
              if(var[i].assigned == false)
                return pch;
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return strdup("InvalidCall");
}


int find4(char *s1)       // function which returns the number of parameters from a given parameters list
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " ,");
  int counter = 0;
  while(pch != NULL)
  {
      counter++;
      pch = strtok (NULL, " ,");
  }
  return counter;
}


char* find5(char *s1)   // function which returns the first undeclared variable from a given expression inside a function (this function is called after the function "check3" returns false)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0)  // not a function
        {
            if(strcmp(pch, "wrong_type_error") != 0)      // we do this to prioritize the error checking
            {
                bool found = false;
                int i;
                for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
                    if(strcmp(pch, fun[currentFunctionIndex].headerVariables[i].name) == 0)
                    {
                      found = true;
                      break;
                    }

                if(found == false)
                    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                        if(strcmp(fun[currentFunctionIndex].localVariables[i].name, pch) == 0)
                        {
                          found = true;
                          break;
                        }

                if(found == false)
                  return pch;
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return strdup("InvalidCall");
}


char* find6(char *s1)     // function which returns the first unassigned variable from a given expression inside a function (this function is called after the function "check4" returns false)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
      {
          if(strcmp(pch, fun[currentFunctionIndex].localVariables[i].name) == 0)
          {
              if(fun[currentFunctionIndex].localVariables[i].assigned == false)
                return pch;
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return strdup("InvalidCall");
}


bool check1(char *s1)   // function which checks if each variable (one that begins with a small letter and it's not a function) from a given expression was declared
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=^,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0 && strcmp(pch, "true") != 0 && strcmp(pch, "false") != 0 && strcmp(pch, "return") != 0)  // not a function or a small letter keyword
        {
            if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
            {
                bool found = false;
                int i;
                for(i = 0; i < varNo; i++)
                {
                    if(strcmp(pch, var[i].name) == 0)
                    {
                      found = true;
                      break;
                    }
                }
                if(found == false)
                  return false;
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=^,");
  }
  return true;
}


bool check2(char *s1)     // function which checks if each variable from a given expression was assigned to a value, and returns false if not
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < varNo; i++)
      {
          if(strcmp(pch, var[i].name) == 0)
          {
              if(var[i].assigned == false)
                return false;
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return true;
}


bool check3(char *s1)     // function which checks if each variable from a given expression (which is inside a function) is declared as a header or local variable in the current function (indicated by a global index)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0)  // not a function
        {
            if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
            {
                int i;
                bool taken = false;
                for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
                    if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, pch) == 0)
                    {
                        taken = true;
                        break;
                    }
                if(taken == false)
                    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                        if(strcmp(fun[currentFunctionIndex].localVariables[i].name, pch) == 0)
                        {
                            taken = true;
                            break;
                        }
                if(taken == false)
                    return false;   // the given variable doesn't exist among the header and local variables of the current function
            }
        }
      }

      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return true;    // all the variables from the given expression are declared inside the current function
}


bool check4(char *s1)     // function which checks if each variable from a given expression (which is inside a function) is assigned as a header or local variable in the current function (indicated by a global index)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      bool taken = false;
      for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
          if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, pch) == 0)
          {
              taken = true;     // if it's a header variable, it's already assigned
              break;
          }
      if(taken == false)
          for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
              if(strcmp(fun[currentFunctionIndex].localVariables[i].name, pch) == 0)
              {
                  if(fun[currentFunctionIndex].localVariables[i].assigned == false)
                      return false;
                  break;
              }

      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return true;    // all the variables from the given expression are assigned inside the current function
}


bool check5(char *s1)     // function which checks if the given variable corresponds to a header/local variable of the current function and if its type is the same as the current function's return type
{
  bool found = false;
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, s1) == 0)
    {
      found = true;
      break;
    }
  if(found == true)
  {
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].type, fun[currentFunctionIndex].returnType) != 0 || fun[currentFunctionIndex].headerVariables[i].length != fun[currentFunctionIndex].returnSize)
      return false;
    return true;
  }

  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
    if(strcmp(fun[currentFunctionIndex].localVariables[i].name, s1) == 0)
    {
      found = true;
      break;
    }
  if(found == true)
  {
    if(strcmp(fun[currentFunctionIndex].localVariables[i].type, fun[currentFunctionIndex].returnType) != 0 || fun[currentFunctionIndex].localVariables[i].length != fun[currentFunctionIndex].returnSize)
      return false;
    return true;
  }

  return false;
}


bool check6(char *s1)     // function which checks if the phrase "length_mismatch_error" is found within a string (for error checking of length mismatch operations)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(strcmp(pch, "length_mismatch_error") == 0)
          return true;
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return false;    // there is no length mismatch error contained in the given expression
}


bool check7(char *s1)    // function which checks if the phrase "wrong_type_error" is found within a string (for error checking of operations between variables)
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      if(strcmp(pch, "wrong_type_error") == 0)
          return true;
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  return false;
}


char* getType(char *s1)     // function which returns the type of a variable, based on its name
{
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
        return var[i].type;
  }
  return strdup("NoType");
}


int getIndex1(char *s1)     // function which returns the index of a variable, based on its name
{
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
        return i;
  }
  return -1;
}


int getIndex2(char *s1)     // function which returns the index of a function, based on its name
{
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name) == 0)
        return i;
  }
  return -1;
}


double getSize1(char *s1)     // function which returns the size of a variable, based on its name
{
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
        return var[i].length;
  }
  return -1;
}


double getSize2(char *s1)     // function which returns the return size of a function, based on its name
{
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name) == 0)
        return fun[i].returnSize;
  }
  return -1;
}


double getSize3(char *s1)     // function which returns the size of a header/local variable inside the current function, based on its name
{
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].headerVariables[i].name) == 0)
        return fun[currentFunctionIndex].headerVariables[i].length;
  }
  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].localVariables[i].name) == 0)
        return fun[currentFunctionIndex].localVariables[i].length;
  }
  return -1;
}


char* extract(char *s1)   // function which receives the start of a function declaration and returns its name
{
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " =");
  return pch;
}


char *structureToString(char *st)     // function used to transfer a string within a structure to a separate memory address (of its own)
{
    char *strA = (char*)malloc(sizeof(char)*strlen(st));
    int i;
    for(i = 0; i < strlen(st); i++)
        strA[i] = st[i];
    strA[strlen(st)] = '\0';
    return strA;
}


















char* singular_declaration_function(char *st1, char *st2)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' exists in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = 1;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
                    finalString[0] = '\0';
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
                var[varNo++].name = strdup(st1);
                var[varNo-1].type = strdup(st2);
                var[varNo-1].length = 1;
                var[varNo-1].assigned = false;
                finalString[0] = '\0';
        }
    }

    return finalString;
}


char* plural_declaration_function(char *st1, char *st2)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = ANY;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
                    finalString[0] = '\0';
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
              var[varNo++].name = strdup(st1);
              var[varNo-1].type = strdup(st2);
              var[varNo-1].length = ANY;
              var[varNo-1].assigned = false;
              finalString[0] = '\0';
        }
    }

    return finalString;
}


char* extended_plural_declaration_function(char *st1, char *st2, char *st3, double d1)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                  if(check7(st3) == true)
                      sprintf(finalString, "error at line %d: There is a wrong used variable contained in the ON expression of the %s variable '%s'", currentLineNumber, st2, st1);
                  else
                  {
                      if(check3(st3) == false)
                          sprintf(finalString, "error at line %d: The variable '%s' contained in the ON expression of the %s variable '%s' is undeclared", currentLineNumber, find5(st3), st2, st1);
                      else
                      {
                          if(check4(st3) == false)
                              sprintf(finalString, "error at line %d: The variable '%s' contained in the ON expression of the %s variable '%s' is unassigned", currentLineNumber, find6(st3), st2, st1);
                          else
                          {
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = d1;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
                              finalString[0] = '\0';
                          }
                      }
                  }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
              if(check7(st3) == true)
                  sprintf(finalString, "error at line %d: There is a wrong used variable contained in the ON expression of the %s variable '%s'", currentLineNumber, st2, st1);
              else
              {
                  if(check1(st3) == false)
                      sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                  else
                  {
                      if(check2(st3) == false)
                          sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                      else
                      {
                          var[varNo++].name = strdup(st1);
                          var[varNo-1].type = strdup(st2);
                          var[varNo-1].length = d1;
                          var[varNo-1].assigned = false;
                          finalString[0] = '\0';
                      }
                  }
              }
        }
    }

    return finalString;
}


char* singular_assignment_function(char *st1, char *st2, char *st3, char *st4)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' from the header of the function '%s' cannot be assigned", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                      sprintf(finalString, "error at line %d: The local %s variable '%s' is reassigned in the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                  else
                  {
                      if(check6(st3) == true)
                          sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                      else
                      {
                          if(find1(st3, st1))
                              sprintf(finalString, "error at line %d: The %s variable '%s' from the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                          else
                          {
                              if(check3(st3) == false)
                                  sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                              else
                              {
                                  if(check4(st3) == false)
                                      sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                                  else
                                  {
                                      if(check7(st3) == true)
                                          sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                      else
                                      {
                                          fun[currentFunctionIndex].localVariables[i].assigned = true;
                                          sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                      }
                                  }
                              }
                          }
                      }
                  }
              else
              {   // deduced declaration
                  if(check6(st3) == true)
                      sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                  else
                  {
                      if(find1(st3, st1))
                          sprintf(finalString, "error at line %d: The %s variable '%s' from the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                      else
                      {
                          if(check3(st3) == false)
                              sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                          else
                          {
                              if(check4(st3) == false)
                                  sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                              else
                              {
                                  if(check7(st3) == true)
                                      sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                  else
                                  {
                                      sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = 1;
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                  }
                              }
                          }
                      }
                  }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is reassigned", currentLineNumber, st2, st1);
              else
              {
                    if(check6(st3) == true)
                        sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                    else
                    {
                        if(find1(st3, st1))
                            sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                        else
                        {
                            if(check1(st3) == false)
                                sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                            else
                            {
                                if(check2(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                                else
                                {
                                    if(check7(st3) == true)
                                        sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                    else
                                    {
                                        var[i].assigned = true;
                                        sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                    }
                                }
                            }
                        }
                    }
              }
        else
        {
            if(check6(st3) == true)
                sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
            else
            {
                if(find1(st3, st1))
                    sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                else
                {
                    if(check1(st3) == false)
                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                    else
                    {
                        if(check2(st3) == false)
                            sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                        else
                        {
                            if(check7(st3) == true)
                                sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                            else
                            {
                                sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                var[varNo++].name = strdup(st1);
                                var[varNo-1].type = strdup(st2);
                                var[varNo-1].length = 1;
                                var[varNo-1].assigned = true;
                            }
                        }
                    }
                }
            }
        }
    }

    return finalString;
}


char* plural_assignment_function(char *st1, char *st2, char *st3, char *st4, double d1)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' from the header of the function '%s' cannot be assigned", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                      sprintf(finalString, "error at line %d: The local %s variable '%s' is reassigned in the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                  else
                  {
                        if(check6(st3) == true)
                            sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                        else
                        {
                            if(find1(st3, st1))
                                sprintf(finalString, "error at line %d: The %s variable '%s' from the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                            else
                            {
                                if(check3(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                                else
                                {
                                    if(check4(st3) == false)
                                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                                    else
                                    {
                                        if(check7(st3) == true)
                                            sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                        else
                                        {
                                            if(getSize3(st1) != d1)
                                                if(getSize3(st1) == ANY)
                                                {
                                                    fun[currentFunctionIndex].localVariables[i].length = d1;
                                                    fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                    sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                                }
                                                else
                                                    sprintf(finalString, "error at line %d: The length of the %s variable '%s' from its definition differs than the length of its assignment in the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                                            else
                                            {
                                                sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                                fun[currentFunctionIndex].localVariables[i].assigned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                  }
              else
              {   // deduced declaration
                  if(check6(st3) == true)
                      sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                  else
                  {
                      if(find1(st3, st1))
                          sprintf(finalString, "error at line %d: The %s variable '%s' from the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                      else
                      {
                          if(check3(st3) == false)
                              sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                          else
                          {
                              if(check4(st3) == false)
                                  sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                              else
                              {
                                  if(check7(st3) == true)
                                      sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                  else
                                  {
                                      sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = d1;
                                      fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                  }
                              }
                          }
                      }
                  }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is reassigned", currentLineNumber, st2, st1);
              else
              {
                    if(check6(st3) == true)
                        sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                    else
                    {
                        if(find1(st3, st1))
                            sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                        else
                        {
                            if(check1(st3) == false)
                                sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                            else
                            {
                                if(check2(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                                else
                                {
                                    if(check7(st3) == true)
                                        sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                    else
                                    {
                                        if(getSize1(st1) != d1)
                                            if(getSize1(st1) == ANY)
                                            {
                                                var[i].length = d1;
                                                var[i].assigned = true;
                                                sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                            }
                                            else
                                                sprintf(finalString, "error at line %d: The length of the %s variable '%s' from its definition differs than the length of its assignment", currentLineNumber, st2, st1);
                                        else
                                        {
                                            sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                            var[i].assigned = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
              }
        else
        {
            if(check6(st3) == true)
                sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
            else
            {
                if(find1(st3, st1))
                    sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                else
                {
                    if(check1(st3) == false)
                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                    else
                    {
                        if(check2(st3) == false)
                            sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                        else
                        {
                              if(check7(st3) == true)
                                  sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                              else
                              {
                                  sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                  var[varNo++].name = strdup(st1);
                                  var[varNo-1].type = strdup(st2);
                                  var[varNo-1].length = d1;
                                  var[varNo-1].assigned = true;
                              }
                        }
                    }
                }
            }
        }
    }

    return finalString;
}


char* singular_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' exists in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                    if(check6(st3) == true)
                        sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                    else
                    {
                        if(find1(st3, st1))
                            sprintf(finalString, "error at line %d: The %s variable '%s' of the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                        else
                        {
                            if(check3(st3) == false)
                                sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                            else
                            {
                                if(check4(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                                else
                                {
                                    if(check7(st3) == true)
                                        sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                    else
                                    {
                                        sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = 1;
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                    }
                                }
                            }
                        }
                    }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
            if(check6(st3) == true)
                sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
            else
            {
                if(find1(st3, st1))
                    sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                else
                {
                    if(check1(st3) == false)
                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                    else
                    {
                        if(check2(st3) == false)
                            sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                        else
                        {
                              if(check7(st3) == true)
                                  sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                              else
                              {
                                  sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                  var[varNo++].name = strdup(st1);
                                  var[varNo-1].type = strdup(st2);
                                  var[varNo-1].length = 1;
                                  var[varNo-1].assigned = true;
                              }
                        }
                    }
                }
            }
        }
    }

    return finalString;
}


char* plural_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4, double d1)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' exists in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                    if(check6(st3) == true)
                        sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                    else
                    {
                        if(find1(st3, st1))
                            sprintf(finalString, "error at line %d: The %s variable '%s' of the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                        else
                        {
                            if(check3(st3) == false)
                                sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                            else
                            {
                                if(check4(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                                else
                                {
                                    if(check7(st3) == true)
                                        sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                    else
                                    {
                                        sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = d1;
                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                    }
                                }
                            }
                        }
                    }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
            if(check6(st3) == true)
                sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
            else
            {
                if(find1(st3, st1))
                    sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                else
                {
                    if(check1(st3) == false)
                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                    else
                    {
                        if(check2(st3) == false)
                            sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                        else
                        {
                              if(check7(st3) == true)
                                  sprintf(finalString, "error at line %d: There is a wrong used variable contained in the assignment expression of the %s variable '%s'", currentLineNumber, st2, st1);
                              else
                              {
                                  sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                  var[varNo++].name = strdup(st1);
                                  var[varNo-1].type = strdup(st2);
                                  var[varNo-1].length = d1;
                                  var[varNo-1].assigned = true;
                              }
                        }
                    }
                }
            }
        }
    }

    return finalString;
}


char* extended_plural_declaration_with_assignment_function(char *st1, char *st2, char *st3, char *st4, char *st5, double d1, double d2)
{
    char finalString[1024];
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name, st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' exists in the header of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared as a local variable of the function '%s'", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
              else
              {
                    if(check6(st3) == true)
                        sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
                    else
                    {
                        if(find1(st3, st1))
                            sprintf(finalString, "error at line %d: The %s variable '%s' of the function '%s' is included in its definition", currentLineNumber, st2, st1, fun[currentFunctionIndex].name);
                        else
                        {
                            if(check3(st3) == false)
                                sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is undeclared", currentLineNumber, find5(st3), st2, st1, fun[currentFunctionIndex].name);
                            else
                            {
                                if(check4(st3) == false)
                                    sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' from the function '%s' is unassigned", currentLineNumber, find6(st3), st2, st1, fun[currentFunctionIndex].name);
                                else
                                {
                                    if(check3(st5) == false)
                                        sprintf(finalString, "error at line %d: The variable '%s' contained in the ON expression of the %s variable '%s' is undeclared", currentLineNumber, find5(st5), st2, st1);
                                    else
                                    {
                                        if(check4(st5) == false)
                                            sprintf(finalString, "error at line %d: The variable '%s' contained in the ON expression of the %s variable '%s' is unassigned", currentLineNumber, find6(st5), st2, st1);
                                        else
                                        {
                                            if(check7(st5) == true)
                                                sprintf(finalString, "error at line %d: There is a wrong used variable contained in the ON expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                            else
                                            {
                                                if(d2 != d1)
                                                    sprintf(finalString, "error at line %d: The length of the %s variable '%s' from its definition differs than the length of its assignment", currentLineNumber, st2, st1);
                                                else
                                                {
                                                    sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = strdup(st1);
                                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = strdup(st2);
                                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = d1;
                                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name, st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
            sprintf(finalString, "error at line %d: The %s variable '%s' is redeclared", currentLineNumber, st2, st1);
        else
        {
            if(check6(st3) == true)
                sprintf(finalString, "error at line %d: Length mismatch found between two terms of an operation", currentLineNumber);
            else
            {
                if(find1(st3, st1))
                    sprintf(finalString, "error at line %d: The %s variable '%s' is included in its definition", currentLineNumber, st2, st1);
                else
                {
                    if(check1(st3) == false)
                        sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st3), st2, st1);
                    else
                    {
                        if(check2(st3) == false)
                            sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st3), st2, st1);
                        else
                        {
                              if(check1(st5) == false)
                                  sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is undeclared", currentLineNumber, find2(st5), st2, st1);
                              else
                              {
                                  if(check2(st5) == false)
                                      sprintf(finalString, "error at line %d: The variable '%s' contained in the definition of the %s variable '%s' is unassigned", currentLineNumber, find3(st5), st2, st1);
                                  else
                                  {
                                      if(check7(st5) == true)
                                          sprintf(finalString, "error at line %d: There is a wrong used variable contained in the ON expression of the %s variable '%s'", currentLineNumber, st2, st1);
                                      else
                                      {
                                          if(d2 != d1)
                                              sprintf(finalString, "error at line %d: The length of the %s variable '%s' from its definition differs than the length of its assignment", currentLineNumber, st2, st1);
                                          else
                                          {
                                              sprintf(finalString, "const %s %s = %s;", st4, st1, st3);
                                              var[varNo++].name = strdup(st1);
                                              var[varNo-1].type = strdup(st2);
                                              var[varNo-1].length = d1;
                                              var[varNo-1].assigned = true;
                                          }
                                      }
                                  }
                              }
                        }
                    }
                }
            }
        }
    }

    return finalString;
}
