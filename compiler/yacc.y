%token INTEGER
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
%token AREA
%token VOLUME
%token NORMAL
%token DOT
%token LENGTH
%token EUCLIDEAN_LENGTH
%token CENTROID
%token GRADIENT
%token DIVERGENCE
%token VARIABLE
%token TRUE
%token FALSE
%token OR
%token AND
%token NOT
%token XOR
%token CEIL
%token FLOOR
%token ABS
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
%token TUPLE
%token USS
%token USSWD
%token USCOS
%token OUTPUT


%start pr
%error-verbose


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

#include <iostream>
#include <cstdio>
#include <sstream>
#include <string>
#include <list>
#include <cstring>
#ifndef _MSC_VER
  #define HEAP_CHECK()
#else
  #include <crtdbg.h>
  #define HEAP_CHECK() _ASSERTE(_CrtCheckMemory())
#endif

using namespace std;

void yyerror(const char* s);
int yylex(void);
bool find1(char* s1, char* s2);
char* find2(char* s1);
char* find3(char* s1);
int find4(char* s1);
char* find5(char* s1);
char* find6(char* s1);
bool check1(char* s1);
bool check2(char* s1);
bool check3(char* s1);
bool check4(char* s1);
bool check5(char* s1);
bool check6(char* s1);
bool check7(char* s1);
bool check8(char* s1, char* s2);
string check9(char* s1);
char* getType(char* s1);
int getIndex1(char* s1);
int getIndex2(char* s1);
double getSize1(char* s1);
double getSize2(char* s1);
double getSize3(char* s1);
int getSize4(char* s1);
char* extract(char* s1);
string CPPToEquelle1(char* st);
double CPPToEquelle2(char* st);
char* EquelleToCPP(string st);
string errorTypeToErrorMessage(string errorType);
string functionToAnySingularType(char *st1, char *st2, char *st3, const string &st4);
string functionToAnyCollectionType(char *st1, char *st2, char *st3, const string &st4);
string singular_declaration_function(char* st1, char* st2);
string plural_declaration_function(char* st1, char* st2);
string extended_plural_declaration_function(char* st1, char* st2, char* st3, double d1);
string singular_assignment_function(char* st1, char* st2, char* st3, char* st4);
string plural_assignment_function(char* st1, char* st2, char* st3, char* st4, double d1);
string singular_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4);
string plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, double d1);
string extended_plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, char* st5, double d1, double d2);
string USS_assignment_function(char* st1);
string USS_declaration_with_assignment_function(char* st1);
string USSWD_assignment_function(char* st1, char* st2);
string USSWD_declaration_with_assignment_function(char* st1, char* st2);
string USCOS_assignment_function(char* st1, char* st2, double d1);
string USCOS_declaration_with_assignment_function(char* st1, char* st2, double d1);
string USCOS_extended_declaration_with_assignment_function(char* st1, char* st2, char* st3, double d1, double d2);
string output_function(char* st1);




// global structure and counter for storing the names of the variables of each type (used for stopping variables reassignment)
struct VariableStructure
{
  string name;       // must begin with a small letter
  string type;       // can be: scalar, vector, vertex, scalars etc.
  double length;     // if the type is a singular type, then the length is 1; otherwise it can be any other number >= 1
  bool assigned;     // we want to know if a variable has been assigned, in order to prevent errors (example: operations with unassigned variables)
};
VariableStructure var[10000];

int varNo = 0;


// global structure and counter for storing the names of the functions
struct FunctionStructure
{
  string name;                                // g1
  string returnType;                          // Collection Of Scalars
  double returnSize;                          // 8
  string paramList;                           // (Cell, Face, CollOfVectors, CollOfScalars On AllFaces(Grid))
  VariableStructure headerVariables[100];     // (c1, f1, pv1, ps1)
  int noParam;                                // 4
  VariableStructure localVariables[100];      // var1, var2, var3
  string signature;                           // (Cell c1, Face f1, CollOfVectors pv1, CollOfScalars On AllFaces(Grid) ps1)
  int noLocalVariables;                       // 3
  bool assigned;              // false
};
FunctionStructure fun[10000];

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

// by summing any multiples of the below values between them, we cannot obtain another unique value (except for: Interior + Boundary = All)

// CONSTANTS
#define INTERIORCELLS     1.01
#define BOUNDARYCELLS     1.02
#define ALLCELLS          2.03
#define INTERIORFACES     1.04
#define BOUNDARYFACES     1.05
#define ALLFACES          2.09
#define INTERIOREDGES     1.10
#define BOUNDARYEDGES     1.11
#define ALLEDGES          2.21
#define INTERIORVERTICES  1.22
#define BOUNDARYVERTICES  1.23
#define ALLVERTICES       2.45
#define ANY               2.46      // the default length of a collection, if it is not explicitly specified


// MACROS
#define STREAM_TO_DOLLARS_CHAR_ARRAY(dd, streamcontent)                 do { stringstream ss; ss << streamcontent; dd = strdup(ss.str().c_str()); } while (false)
#define LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY(dd)           do { stringstream ss; ss << "length_mismatch_error"; dd = strdup(ss.str().c_str()); } while (false)
// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)                          do { stringstream ss; ss << "wrong_type_error  " << d1; dd = strdup(ss.str().c_str()); }  while (false)
// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)            do { stringstream ss; ss << "wrong_type_error  " << d1; dd = strdup(ss.str().c_str()); }  while (false)


%}


%type<str> floating_point
%type<str> number
%type<str> scalar_expr
%type<inf> scalar_exprs
%type<str> scalar_term
%type<inf> scalar_terms
%type<str> scalar_factor
%type<inf> scalar_factors
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
%type<str> INTEGER
%type<str> VARIABLE
%type<str> COMMENT
%type<inf> plural
%type<str> header
%type<str> parameter_list
%type<inf> type

// option 1
/*
%type<inf> value
%type<arr> values
*/

// option 2
/*
%type<str> value
%type<dinf> values
*/

// option 3
%type<str> values

%type<str> end_lines
%type<inf> return_instr
%type<str> function_start
%type<str> function_declaration
%type<str> function_assignment
%type<str> commands
%type<str> command
%type<str> command1;
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
%type<str> output





%code requires
{
  struct info
  {
      double size;
      char* str;
  };

  // struct array
  // {
  //     char *sizes;
  //     char *str;
  // };

  struct dinfo
  {
      char* cCode;
      char* sepCode;
  };
}



%union
{
  int value;
  char* str;           // the non-terminals which need to store only the translation code for C++ will be declared with this type
  struct info inf;            // the non-terminals which need to store both the translation code for C++ and the size of the collection will be declared with this type
  //struct array arr;  // the values which are passed as arguments when calling a function must be checked to see if they correspond to the function's template
  struct dinfo dinf;
};


%%


floating_point: INTEGER '.' INTEGER          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << "." << $3); }
              ;


number: INTEGER                              { $$ = strdup($1); }
      | floating_point                       { $$ = strdup($1); }
      ;


scalar_expr: scalar_term                     { $$ = strdup($1); }
           | '-' scalar_term                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "-" << $2); }
           | scalar_expr '+' scalar_term     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " + " << $3); }
           | scalar_expr '-' scalar_term     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " - " << $3); }
           ;


scalar_term: scalar_factor                           { $$ = strdup($1); }
           | scalar_term '*' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " * " << $3); }
           | scalar_factor '*' scalar_term           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " * " << $3); }
           | scalar_term '/' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " / " << $3); }
           | scalar_term '^' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.pow(" << $1 << ", " << $3 << ")"); }
           ;


scalar_factor: number                                  { $$ = strdup($1); }
             | '(' scalar_expr ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "(" << $2 << ")"); }
             | EUCLIDEAN_LENGTH '(' vector_expr ')'    { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.euclideanLength(" << $3 << ")"); }
             | LENGTH '(' edge ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.length(" << $3 << ")"); }
             | AREA '(' face ')'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.area(" << $3 << ")"); }
             | VOLUME '(' cell ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.volume(" << $3 << ")"); }
             | DOT '(' vector_expr ',' vector_expr ')' { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.dot(" << $3 << ", " << $5 << ")"); }
             | CEIL '(' scalar_expr ')'                { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.ceil(" << $3 << ")"); }
             | FLOOR '(' scalar_expr ')'               { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.floor(" << $3 << ")"); }
             | ABS '(' scalar_expr ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.abs(" << $3 << ")"); }
             | MIN '(' scalars ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.min(" << $3.str << ")"); }
             | MAX '(' scalars ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.max(" << $3.str << ")"); }
             | GRADIENT '(' scalar_expr ')'            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.gradient(" << $3 << ")"); }
             | DIVERGENCE '(' scalar_expr ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.divergence(" << $3 << ")"); }
             | VARIABLE                                {
                                                          if(strcmp(getType($1), "scalar") != 0)
                                                          {
                                                              WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                                          }
                                                          else
                                                          {
                                                              $$ = strdup($1);
                                                          }
                                                       }

             | VARIABLE '(' values ')'                 {
                                                          string st = functionToAnySingularType($1, "Scalar", $3, "scalar");
                                                          $$ = strdup(st.c_str());
                                                       }
             ;


scalars: scalar_exprs                 { $$.str = strdup($1.str); $$.size = $1.size; }
       | scalars ',' scalar_exprs     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); $$.size = $1.size + $3.size; }
       | scalar_expr                  { $$.str = strdup($1); $$.size = 1; }
       | scalars ',' scalar_expr      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3); $$.size = $1.size + 1; }
       ;


scalar_exprs: scalar_terms                     { $$.str = strdup($1.str); $$.size = $1.size; }
            | '-' scalar_terms                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); $$.size = $2.size; }
            | scalar_exprs '+' scalar_terms
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str);
                                                      $$.size = $1.size;
                                                  }
                                               }

            | scalar_exprs '-' scalar_terms
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            ;


scalar_terms: scalar_factors                    { $$.str = strdup($1.str); $$.size = $1.size; }
            | scalar_terms '*' scalar_factors
                                                {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_factors '*' scalar_terms
                                                {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_terms '/' scalar_factors
                                               {
                                                  if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " / " << $3.str);
                                                      $$.size = $1.size;
                                                  }
                                               }
            | scalar_terms '^' scalar_factor   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.pow(" << $1.str << ", " << $3 << ")"); $$.size = $1.size; }
            ;


scalar_factors: EUCLIDEAN_LENGTH '(' vector_exprs ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.euclideanLength(" << $3.str << ")"); $$.size = $3.size; }
              | LENGTH '(' edges ')'                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.length(" << $3.str << ")"); $$.size = $3.size; }
              | AREA '(' faces ')'                              { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.area(" << $3.str << ")"); $$.size = $3.size; }
              | VOLUME '(' cells ')'                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.volume(" << $3.str << ")"); $$.size = $3.size; }
              | DOT '(' vector_exprs ',' vector_exprs ')'
                                                                {
                                                                   if($3.size != $5.size)    // check that the lengths of the 2 terms are equal
                                                                   {
                                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                                  }
                                                                   else
                                                                   {
                                                                       STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.dot(" << $3.str << ", " << $5.str << ")");
                                                                       $$.size = $3.size;
                                                                   }
                                                                }

              | CEIL '(' scalar_exprs ')'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.ceil(" << $3.str << ")"); $$.size = $3.size;}
              | FLOOR '(' scalar_exprs ')'                      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.floor(" << $3.str << ")"); $$.size = $3.size;}
              | ABS '(' scalar_exprs ')'                        { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.abs(" << $3.str << ")"); $$.size = $3.size;}
              | GRADIENT '(' scalar_exprs ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); $$.size = $3.size; }
              | DIVERGENCE '(' scalar_exprs ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); $$.size = $3.size;}
              | VARIABLE                                        {
                                                                    if(strcmp(getType($1), "scalars") != 0)
                                                                    {
                                                                        WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                                                    }
                                                                    else
                                                                    {
                                                                        $$.str = strdup($1);
                                                                        $$.size = getSize1($1);
                                                                    }
                                                                }

              | VARIABLE '(' values ')'                         {
                                                                    string st = functionToAnyCollectionType($1, "CollOfScalars", $3, "scalars");
                                                                    if(st == "ok")
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                                        $$.size = fun[getIndex2($1)].returnSize;
                                                                    }
                                                                    else
                                                                    {
                                                                        $$.str = strdup(st.c_str());
                                                                    }
                                                                }
              ;


vector_expr: vector_term                      { $$ = strdup($1); }
           | '-' vector_term                  { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "-" << $2); }
           | vector_expr '+' vector_term      { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " + " << $3); }
           | vector_expr '-' vector_term      { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " - " << $3); }
           ;


vector_term: '[' scalars ']'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "[" << $2.str << "]"); }
           | CENTROID '(' cell ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.centroid(" << $3 << ")"); }
           | CENTROID '(' face ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.centroid(" << $3 << ")"); }
           | NORMAL '(' face ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.normal(" << $3 << ")"); }
           | '(' vector_expr ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "(" << $2 << ")"); }              // produces 1 shift/reduce conflict
           | vector_term '*' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " * " << $3); }             // produces 1 reduce/reduce conflict
           | scalar_factor '*' vector_term         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " * " << $3); }
           | vector_term '/' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " / " << $3); }
           | VARIABLE                              {
                                                      if(strcmp(getType($1), "vector") != 0)
                                                      {
                                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                                      }
                                                      else
                                                      {
                                                          $$ = strdup($1);
                                                      }
                                                   }

           | VARIABLE '(' values ')'               {
                                                      string st = functionToAnySingularType($1, "Vector", $3, "vector");
                                                      $$ = strdup(st.c_str());
                                                   }
           ;


vectors: vector_term                      { $$.str = strdup($1); $$.size = 1; }
       | vectors ',' vector_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3); $$.size = $1.size + 1; }
       ;


vector_exprs: vector_terms                       { $$.str = strdup($1.str); $$.size = $1.size; }
            | '-' vector_terms                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); $$.size = $2.size; }            // produces 1 shift/reduce conflict
            | vector_exprs '+' vector_terms
                                                 {
                                                    if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                    {
                                                        LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str);
                                                        $$.size = $1.size;
                                                    }
                                                 }

            | vector_exprs '-' vector_terms
                                                 {
                                                    if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                    {
                                                        LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str);
                                                        $$.size = $1.size;
                                                    }
                                                 }
            ;


vector_terms: '[' vectors ']'                        { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "[" << $2.str << "]"); $$.size = $2.size; }
            | CENTROID '(' cells ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); $$.size = $3.size; }
            | CENTROID '(' faces ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); $$.size = $3.size; }
            | NORMAL '(' faces ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.normal(" << $3.str << ")"); $$.size = $3.size; }
            | '(' vector_exprs ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); $$.size = $2.size; }          // produces 1 shift/reduce conflict
            | vector_terms '*' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3); $$.size = $1.size; }         // produces 1 reduce/reduce conflict
            | scalar_factor '*' vector_terms         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << " - " << $3.str); $$.size = $3.size; }
            | vector_terms '/' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3); $$.size = $1.size; }
            | VARIABLE                               {
                                                        if(strcmp(getType($1), "vectors") != 0)
                                                        {
                                                            WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                                        }
                                                        else
                                                        {
                                                            $$.str = strdup($1);
                                                            $$.size = getSize1($1);
                                                        }
                                                     }

            | VARIABLE '(' values ')'                {
                                                          string st = functionToAnyCollectionType($1, "CollOfVectors", $3, "vectors");
                                                          if(st == "ok")
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                              $$.size = fun[getIndex2($1)].returnSize;
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup(st.c_str());
                                                          }
                                                     }
            ;


vertex: VARIABLE           {
                              if(strcmp(getType($1), "vertex") != 0)
                              {
                                  WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                              }
                              else
                              {
                                  $$ = strdup($1);
                              }
                           }

      | VARIABLE '(' values ')'               {
                                                  string st = functionToAnySingularType($1, "Vertex", $3, "vertex");
                                                  $$ = strdup(st.c_str());
                                              }
      ;


vertices: INTERIOR_VERTICES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorVertices()"); $$.size = INTERIORVERTICES; }
        | BOUNDARY_VERTICES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryVertices()"); $$.size = BOUNDARYVERTICES; }
        | ALL_VERTICES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allVertices()"); $$.size = ALLVERTICES; }
        | VARIABLE                            {
                                                  if(strcmp(getType($1), "vertices") != 0)
                                                  {
                                                      WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                                  }
                                                  else
                                                  {
                                                      $$.str = strdup($1);
                                                      $$.size = getSize1($1);
                                                  }
                                              }

        | VARIABLE '(' values ')'                 {
                                                      string st = functionToAnyCollectionType($1, "CollOfVertices", $3, "vertices");
                                                      if(st == "ok")
                                                      {
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                          $$.size = fun[getIndex2($1)].returnSize;
                                                      }
                                                      else
                                                      {
                                                          $$.str = strdup(st.c_str());
                                                      }
                                                  }
        ;


edge: VARIABLE             {
                              if(strcmp(getType($1), "edge") != 0)
                              {
                                  WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                              }
                              else
                              {
                                  $$ = strdup($1);
                              }
                           }

    | VARIABLE '(' values ')'               {
                                                string st = functionToAnySingularType($1, "Edge", $3, "edge");
                                                $$ = strdup(st.c_str());
                                            }
    ;


edges: INTERIOR_EDGES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorEdges()"); $$.size = INTERIOREDGES; }
     | BOUNDARY_EDGES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryEdges()"); $$.size = BOUNDARYEDGES; }
     | ALL_EDGES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allEdges()"); $$.size = ALLEDGES; }
     | VARIABLE                         {
                                            if(strcmp(getType($1), "edges") != 0)
                                            {
                                                WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1);
                                                $$.size = getSize1($1);
                                            }
                                        }

     | VARIABLE '(' values ')'                  {
                                                    string st = functionToAnyCollectionType($1, "CollOfEdges", $3, "edges");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                        $$.size = fun[getIndex2($1)].returnSize;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
     ;


face: VARIABLE                    {
                                      if(strcmp(getType($1), "face") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }

    | VARIABLE '(' values ')'               {
                                                string st = functionToAnySingularType($1, "Face", $3, "face");
                                                $$ = strdup(st.c_str());
                                            }
    ;


faces: INTERIOR_FACES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorFaces()"); $$.size = INTERIORFACES; }
     | BOUNDARY_FACES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryFaces()"); $$.size = BOUNDARYFACES; }
     | ALL_FACES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allFaces()"); $$.size = ALLFACES; }
     | VARIABLE                         {
                                            if(strcmp(getType($1), "faces") != 0)
                                            {
                                                WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1);
                                                $$.size = getSize1($1);
                                            }
                                        }

     | VARIABLE '(' values ')'                   {
                                                      string st = functionToAnyCollectionType($1, "CollOfFaces", $3, "faces");
                                                      if(st == "ok")
                                                      {
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                          $$.size = fun[getIndex2($1)].returnSize;
                                                      }
                                                      else
                                                      {
                                                          $$.str = strdup(st.c_str());
                                                      }
                                                 }
     ;


cell: FIRST_CELL '(' face ')'     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.firstCell(" << $3 << ")"); }
    | SECOND_CELL '(' face ')'    { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.secondCell(" << $3 << ")"); }
    | VARIABLE                    {
                                      if(strcmp(getType($1), "cell") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }

    | VARIABLE '(' values ')'                 {
                                                  string st = functionToAnySingularType($1, "Cell", $3, "cell");
                                                  $$ = strdup(st.c_str());
                                              }
    ;


cells: INTERIOR_CELLS '(' GRID ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorCells()"); $$.size = INTERIORCELLS; }
     | BOUNDARY_CELLS '(' GRID ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryCells()"); $$.size = BOUNDARYCELLS;}
     | ALL_CELLS '(' GRID ')'               { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allCells()"); $$.size = ALLCELLS;}
     | FIRST_CELL '(' faces ')'             { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.firstCell(" << $3.str << ")"); $$.size = $3.size;}
     | SECOND_CELL '(' faces ')'            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.secondCell(" << $3.str << ")"); $$.size = $3.size;}
     | VARIABLE                             {
                                                if(strcmp(getType($1), "cells") != 0)
                                                {
                                                    WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                                }
                                                else
                                                {
                                                    $$.str = strdup($1);
                                                    $$.size = getSize1($1);
                                                }
                                            }

     | VARIABLE '(' values ')'                  {
                                                    string st = functionToAnyCollectionType($1, "CollOfCells", $3, "cells");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                        $$.size = fun[getIndex2($1)].returnSize;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
     ;


adb: GRADIENT '(' adb ')'         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.gradient(" << $3 << ")"); }
   | DIVERGENCE '(' adb ')'       { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "er.divergence(" << $3 << ")"); }
   | VARIABLE                     {
                                      if(strcmp(getType($1), "scalarAD") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                      }
                                      else
                                      {
                                          $$ = strdup($1);
                                      }
                                  }

   | VARIABLE '(' values ')'                {
                                                string st = functionToAnySingularType($1, "ScalarAD", $3, "scalarAD");
                                                $$ = strdup(st.c_str());
                                            }
   ;


adbs: GRADIENT '(' adbs ')'       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); $$.size = $3.size; }
    | DIVERGENCE '(' adbs ')'     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); $$.size = $3.size;}
    | VARIABLE                    {
                                      if(strcmp(getType($1), "scalarsAD") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1);
                                          $$.size = getSize1($1);
                                      }
                                  }

    | VARIABLE '(' values ')'                   {
                                                    string st = functionToAnyCollectionType($1, "CollOfScalarsAD", $3, "scalarsAD");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                        $$.size = fun[getIndex2($1)].returnSize;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
    ;


boolean_expr: boolean_term                           { $$ = strdup($1); }
            | NOT boolean_term                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "!" << $2); }
            | boolean_expr AND boolean_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " && " << $3); }
            | boolean_expr OR boolean_term           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " || " << $3); }
            | boolean_expr XOR boolean_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "(!" << $1 << " && " << $3 << ") || (!" << $3 << " && " << $1 << ")"); }
            ;



boolean_term: TRUE                                   { $$ = strdup("true"); }
            | FALSE                                  { $$ = strdup("false"); }
            | scalar_expr '>' scalar_expr            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " > " << $3); }
            | scalar_expr '<' scalar_expr            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " < " << $3); }
            | scalar_expr LESSEQ scalar_expr         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " <= " << $3); }
            | scalar_expr GREATEREQ scalar_expr      { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " >= " << $3); }
            | scalar_expr EQ scalar_expr             { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " == " << $3); }
            | scalar_expr NOTEQ scalar_expr          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " != " << $3); }
            | '(' boolean_expr ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "(" << $2 << ")"); }
            | VARIABLE                               {
                                                        if(strcmp(getType($1), "bool") != 0)
                                                        {
                                                            WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$, $1);
                                                        }
                                                        else
                                                        {
                                                            $$ = strdup($1);
                                                        }
                                                    }

            | VARIABLE '(' values ')'               {
                                                        string st = functionToAnySingularType($1, "bool", $3, "bool");
                                                        $$ = strdup(st.c_str());
                                                    }
            ;


boolean_exprs: boolean_terms                           { $$.str = strdup($1.str); $$.size = $1.size; }
             | NOT boolean_terms                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "!" << $2.str); $$.size = $2.size;}
             | boolean_exprs AND boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " && " << $3.str);
                                                              $$.size = $1.size;
                                                          }
                                                       }

             | boolean_exprs OR boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " || " << $3.str);
                                                              $$.size = $1.size;
                                                          }
                                                       }

             | boolean_exprs XOR boolean_terms
                                                       {
                                                          if($1.size != $3.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(!" << $1.str << " && " << $3.str << ") || (!" << $3.str << " && " << $1.str << ")");
                                                              $$.size = $1.size;
                                                          }
                                                       }
             ;



boolean_terms: '(' scalars ')' '>' '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") > (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' '<' '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") < (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' LESSEQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") <= (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' GREATEREQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") >= (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' EQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") == (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }

             | '(' scalars ')' NOTEQ '(' scalars ')'
                                                      {
                                                          if($2.size != $6.size)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") != (" << $6.str << ")");
                                                              $$.size = $2.size;
                                                          }
                                                      }
             | '(' boolean_exprs ')'                  { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); $$.size = $2.size;}
             | VARIABLE                               {
                                                          if(strcmp(getType($1), "bools") != 0)
                                                          {
                                                              WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1);
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup($1);
                                                              $$.size = getSize1($1);
                                                          }
                                                      }

             | VARIABLE '(' values ')'                {
                                                          string st = functionToAnyCollectionType($1, "CollOfBools", $3, "bools");
                                                          if(st == "ok")
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1 << "(" << $3 << ")");
                                                              $$.size = fun[getIndex2($1)].returnSize;
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup(st.c_str());
                                                          }
                                                      }
             ;



plural: scalar_exprs            { $$.str = strdup($1.str); $$.size = $1.size; }
      | vector_exprs            { $$.str = strdup($1.str); $$.size = $1.size; }
      | vertices                { $$.str = strdup($1.str); $$.size = $1.size; }
      | edges                   { $$.str = strdup($1.str); $$.size = $1.size; }
      | faces                   { $$.str = strdup($1.str); $$.size = $1.size; }
      | cells                   { $$.str = strdup($1.str); $$.size = $1.size; }
      | adbs                    { $$.str = strdup($1.str); $$.size = $1.size; }
      | boolean_exprs           { $$.str = strdup($1.str); $$.size = $1.size; }
      ;


header: VARIABLE HEADER_DECL SCALAR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Scalar " << $1); }
      | VARIABLE HEADER_DECL VECTOR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Vector " << $1); }
      | VARIABLE HEADER_DECL VERTEX                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Vertex " << $1); }
      | VARIABLE HEADER_DECL EDGE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Edge " << $1); }
      | VARIABLE HEADER_DECL FACE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Face " << $1); }
      | VARIABLE HEADER_DECL CELL                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "Cell " << $1); }
      | VARIABLE HEADER_DECL ADB                             { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "ScalarAD " << $1); }
      | VARIABLE HEADER_DECL BOOLEAN                         { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "bool " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF SCALAR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfScalars " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF VECTOR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfVectors " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF VERTEX            { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfVertices " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF EDGE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfEdges " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF FACE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfFaces " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF CELL              { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfCells " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF ADB               { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfScalarsAD " << $1); }
      | VARIABLE HEADER_DECL COLLECTION OF BOOLEAN           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "CollOfBools " << $1); }
      ;


parameter_list: header                         { $$ = strdup($1); }
              | parameter_list ',' header      { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << ", " << $3); }
              ;


commands: command1                              { $$ = strdup($1); }
        | commands end_lines command1           { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << $2 << $3); }
        |                                       { $$ = strdup(""); }     // a function can have only the return instruction
        ;


type: SCALAR                                { $$.str = strdup("Scalar"); $$.size = ANY; }
    | VECTOR                                { $$.str = strdup("Vector"); $$.size = ANY; }
    | VERTEX                                { $$.str = strdup("Vertex"); $$.size = ANY; }
    | EDGE                                  { $$.str = strdup("Edge"); $$.size = ANY; }
    | FACE                                  { $$.str = strdup("Face"); $$.size = ANY; }
    | CELL                                  { $$.str = strdup("Cell"); $$.size = ANY; }
    | ADB                                   { $$.str = strdup("ScalarAD"); $$.size = ANY; }
    | BOOLEAN                               { $$.str = strdup("bool"); $$.size = ANY; }
    | COLLECTION OF SCALAR                  { $$.str = strdup("CollOfScalars"); $$.size = ANY; }
    | COLLECTION OF VECTOR                  { $$.str = strdup("CollOfVectors"); $$.size = ANY; }
    | COLLECTION OF VERTEX                  { $$.str = strdup("CollOfVertices"); $$.size = ANY; }
    | COLLECTION OF EDGE                    { $$.str = strdup("CollOfEdges"); $$.size = ANY; }
    | COLLECTION OF FACE                    { $$.str = strdup("CollOfFaces"); $$.size = ANY; }
    | COLLECTION OF CELL                    { $$.str = strdup("CollOfCells"); $$.size = ANY; }
    | COLLECTION OF ADB                     { $$.str = strdup("CollOfScalarsAD"); $$.size = ANY; }
    | COLLECTION OF BOOLEAN                 { $$.str = strdup("CollOfBools"); $$.size = ANY; }
    | COLLECTION OF SCALAR ON plural        { $$.str = strdup("CollOfScalars"); $$.size = $5.size; }
    | COLLECTION OF VECTOR ON plural        { $$.str = strdup("CollOfVectors"); $$.size = $5.size; }
    | COLLECTION OF VERTEX ON plural        { $$.str = strdup("CollOfVertices"); $$.size = $5.size; }
    | COLLECTION OF EDGE ON plural          { $$.str = strdup("CollOfEdges"); $$.size = $5.size; }
    | COLLECTION OF FACE ON plural          { $$.str = strdup("CollOfFaces"); $$.size = $5.size; }
    | COLLECTION OF CELL ON plural          { $$.str = strdup("CollOfCells"); $$.size = $5.size; }
    | COLLECTION OF ADB ON plural           { $$.str = strdup("CollOfScalarsAD"); $$.size = $5.size; }
    | COLLECTION OF BOOLEAN ON plural       { $$.str = strdup("CollOfBools"); $$.size = $5.size; }
    ;


//////////////////////////////////////////////////////////////////////// these support input parameters as expressions with or without ON (option 1)
/*
value: scalar            {$$.str = strdup($1); $$.size = 1;}
     | vector            {$$.str = strdup($1); $$.size = 1;}
     | vertex            {$$.str = strdup($1); $$.size = 1;}
     | edge              {$$.str = strdup($1); $$.size = 1;}
     | face              {$$.str = strdup($1); $$.size = 1;}
     | cell              {$$.str = strdup($1); $$.size = 1;}
     | adb               {$$.str = strdup($1); $$.size = 1;}
     | boolean           {$$.str = strdup($1); $$.size = 1;}
     | scalar_exprs      {$$.str = strdup($1.size); $$.size = $1.size;}
     | vector_exprs      {$$.str = strdup($1.size); $$.size = $1.size;}
     | vertices          {$$.str = strdup($1.size); $$.size = $1.size;}
     | edges             {$$.str = strdup($1.size); $$.size = $1.size;}
     | faces             {$$.str = strdup($1.size); $$.size = $1.size;}
     | cells             {$$.str = strdup($1.size); $$.size = $1.size;}
     | adbs              {$$.str = strdup($1.size); $$.size = $1.size;}
     | booleans          {$$.str = strdup($1.size); $$.size = $1.size;}
     ;


values: value                   {$$.str = strdup($1.str); itoa($$.sizes, $1.sizes, 100);}
      | values ',' value        {
                                  char *str = append5($1.str,',',$3.str);
                                  $$.str = strdup(str);
                                  free(str);
                                  char *temp = (char *)malloc(1000 * sizeof(char));
                                  itoa(temp, $3.size, 100);
                                  char *str2 = append5($1.sizes,',',temp);
                                  $$.sizes = strdup(str2);
                                  free(str2);
                                }
      ;
*/


//////////////////////////////////////////////////////////////////////// these support input parameters as expressions without ON (option 2)
/*
value: scalar_expr       {$$ = strdup($1);}
     | vector_expr       {$$ = strdup($1);}
     | vertex            {$$ = strdup($1);}
     | edge              {$$ = strdup($1);}
     | face              {$$ = strdup($1);}
     | cell              {$$ = strdup($1);}
     | adb               {$$ = strdup($1);}
     | boolean_expr      {$$ = strdup($1);}
     | scalar_exprs      {$$ = strdup($1.str);}
     | vector_exprs      {$$ = strdup($1.str);}
     | vertices          {$$ = strdup($1.str);}
     | edges             {$$ = strdup($1.str);}
     | faces             {$$ = strdup($1.str);}
     | cells             {$$ = strdup($1.str);}
     | adbs              {$$ = strdup($1.str);}
     | boolean_exprs     {$$ = strdup($1.str);}
     ;


// we need 'values' to be a structure with 2 strings: one which will store the exact output which should be displayed, and another which should store all the terms separated by an unique character ('@')
values: value                   {$$.cCode = strdup($1); $$.sepCode = strdup($1);}
      | values ',' value        {char *str = append5($1.cCode,',',$3); $$.cCode = strdup(str); free(str); $$.sepCode = append5($1.sepCode, '@', $3);}
      ;
*/


//////////////////////////////////////////////////////////////////////// this supports input parameters as variables
values: VARIABLE                { $$ = strdup($1); }
      | values ',' VARIABLE     { STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << ", " << $3); }
      ;


end_lines: '\n'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "\n"); currentLineNumber++; }
         | '\n' end_lines       { STREAM_TO_DOLLARS_CHAR_ARRAY($$, "\n" << $2); currentLineNumber++; }
         |                      { $$ = strdup(""); }
         ;


return_instr: RETURN boolean_expr '?' VARIABLE ':' VARIABLE
                  {
                    if(check5($4) == false || check5($6) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.size = -1;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "return " << $2 << " ? " << $4 << " : " << $6);
                        $$.size = getSize3($4);
                    }
                  }

            | RETURN VARIABLE
                  {
                    if(check5($2) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.size = -1;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "return " << $2 << ";");
                        $$.size = getSize3($2);
                    }
                  }
            ;


function_start: VARIABLE '=' end_lines '{'
                                            {
                                              insideFunction = true;
                                              currentFunctionIndex = getIndex2($1);
                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << " = " << $3 << "{");
                                            }


// these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: VARIABLE ':' FUNCTION '(' parameter_list ')' RET type
                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), $1) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$, "error at line " << currentLineNumber << ": The function '" << $1 << "' is redeclared");
                                                }
                                                else
                                                {
                                                        fun[funNo++].name = strdup($1);
                                                        fun[funNo-1].returnType = CPPToEquelle1($8.str);
                                                        fun[funNo-1].returnSize = $8.size;
                                                        fun[funNo-1].noLocalVariables = 0;
                                                        fun[funNo-1].noParam = 0;

                                                        char *cs1 = strdup($5);    // we need to make a copy, because the strtok function modifies the given string
                                                        char *pch;
                                                        char *pch2;
                                                        char *cs2;
                                                        pch = strtok(cs1, ",");

                                                        while(pch != NULL)
                                                        {
                                                          cs2 = strdup(pch);
                                                          pch2 = strtok(cs2, " ");    // type of the variable

                                                          stringstream ss;
                                                          ss << fun[funNo-1].paramList << ", " << pch2;

                                                          fun[funNo-1].paramList = strdup(ss.str().c_str());
                                                          char *copy = strdup(pch2);
                                                          pch2 = strtok(NULL, " ");   // name of the variable

                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam++].name = strdup(pch2);
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].type = CPPToEquelle1(copy);    // the string we have as a parameter list is already transformed in C++, but we need the types' keywords from Equelle
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].length = CPPToEquelle2(copy);  // the string we have as a parameter list is already transformed in C++, but we need the types' lengths
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].assigned = true;
                                                          fun[funNo-1].signature = strdup($5);

                                                          pch = strtok(NULL, ",");
                                                        }

                                                        fun[funNo-1].assigned = false;
                                                        // STREAM_TO_DOLLARS_CHAR_ARRAY($$, $8.str << " " << $1 << "(" << $5 << ")" << ";");
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$, "");
                                                }
                                            }
                    ;


function_assignment: function_start end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional

                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), extract($1)) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                      if(fun[i].assigned == true)
                                                      {
                                                          stringstream ss;
                                                          ss << "error at line " << currentLineNumber << ": The function '" << fun[i].name << "' is reassigned";
                                                          $$ = strdup(ss.str().c_str());
                                                      }
                                                      else
                                                      {
                                                          if($5.size != -1)
                                                          {
                                                              // STREAM_TO_DOLLARS_CHAR_ARRAY($$, $1 << $2 << $3 << $4 << $5.str << $6 << "}");
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$, "auto " << fun[i].name << "[&](" << fun[i].signature << ") -> " << EquelleToCPP(fun[i].returnType) << " {\n" << $2 << $3 << $4 << $5.str << $6 << "}");
                                                              if(fun[i].returnSize == ANY && $5.size != ANY)
                                                                  fun[i].returnSize = $5.size;
                                                              else
                                                                  if(fun[i].returnSize != ANY && $5.size == ANY)
                                                                      {;}   // do nothing (the function must keep its return size from the definition)
                                                                  else
                                                                      {;}   // if both are ANY, the function's return type is already correct; if none are ANY, then they should already be equal, otherwise the instruction flow wouldn't enter on this branch
                                                              fun[i].assigned = true;
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$, "error at line " << currentLineNumber << ": At least one of the return variables does not exist within the function or the return type of the function '" << fun[i].name << "' from its assignment differs than the length of the return type from the function's definition");
                                                          }

                                                      }
                                                else
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$, "error at line " << currentLineNumber << ": The function '" << extract($1) <<"' must be declared before being assigned");
                                                }

                                                insideFunction = false;
                                                currentFunctionIndex = -1;
                                            }
                   ;




// function_declaration_with_assignment: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type '=' end_lines '{' end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
//                                     ; // tre sa punem booleana globala true inainte sa execute comenzile din functie



/*
tuple_declaration: VARIABLE ':' TUPLE OF '(' type ')'

tuple_assignment: VARIABLE '=' '(' entities ')'

tuple_declaration_with_assignment: VARIABLE ':' TUPLE OF '(' type ')' '=' '(' entities ')'
*/






output: OUTPUT '(' VARIABLE ')'       { string out = output_function($3); $$ = strdup(out.c_str()); }








singular_declaration: VARIABLE ':' SCALAR               { string out = singular_declaration_function($1, "scalar"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' VECTOR               { string out = singular_declaration_function($1, "vector"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' VERTEX               { string out = singular_declaration_function($1, "vertex"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' EDGE                 { string out = singular_declaration_function($1, "edge"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' FACE                 { string out = singular_declaration_function($1, "face"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' CELL                 { string out = singular_declaration_function($1, "cell"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' ADB                  { string out = singular_declaration_function($1, "scalarAD"); $$ = strdup(out.c_str()); }
                    | VARIABLE ':' BOOLEAN              { string out = singular_declaration_function($1, "bool"); $$ = strdup(out.c_str()); }
                    ;


plural_declaration: VARIABLE ':' COLLECTION OF SCALAR       { string out = plural_declaration_function($1, "scalars"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF VECTOR       { string out = plural_declaration_function($1, "vectors"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF VERTEX       { string out = plural_declaration_function($1, "vertices"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF EDGE         { string out = plural_declaration_function($1, "edges"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF FACE         { string out = plural_declaration_function($1, "faces"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF CELL         { string out = plural_declaration_function($1, "cells"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF ADB          { string out = plural_declaration_function($1, "scalarsAD"); $$ = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF BOOLEAN      { string out = plural_declaration_function($1, "bools"); $$ = strdup(out.c_str()); }
                  ;


extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "scalars", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF VECTOR ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "vectors", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF VERTEX ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "vertices", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF EDGE ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "edges", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF FACE ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "faces", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF CELL ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "cells", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF ADB ON plural         { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "scalarsAD", st, $7.size); $$ = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural     { char *st = strdup($7.str); string out = extended_plural_declaration_function($1, "bools", st, $7.size); $$ = strdup(out.c_str()); }
                           ;


declaration: singular_declaration           { char* out = strdup($1); $$ = out; }
           | plural_declaration             { char* out = strdup($1); $$ = out; }
           | extended_plural_declaration    { char* out = strdup($1); $$ = out; }
           ;



singular_assignment: VARIABLE '=' scalar_expr              { char *st = strdup($3); string out = singular_assignment_function($1, "scalar", st, "Scalar"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' vector_expr              { char *st = strdup($3); string out = singular_assignment_function($1, "vector", st, "Vector"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' vertex                   { char *st = strdup($3); string out = singular_assignment_function($1, "vertex", st, "Vertex"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' edge                     { char *st = strdup($3); string out = singular_assignment_function($1, "edge", st, "Edge"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' face                     { char *st = strdup($3); string out = singular_assignment_function($1, "face", st, "Face"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' cell                     { char *st = strdup($3); string out = singular_assignment_function($1, "cell", st, "Cell"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' adb                      { char *st = strdup($3); string out = singular_assignment_function($1, "scalarAD", st, "ScalarAD"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' boolean_expr             { char *st = strdup($3); string out = singular_assignment_function($1, "bool", st, "bool"); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' USS                      { string out = USS_assignment_function($1); $$ = strdup(out.c_str()); }
                   | VARIABLE '=' USSWD '(' number ')'     { string out = USSWD_assignment_function($1, $5); $$ = strdup(out.c_str()); }
                   ;


plural_assignment: VARIABLE '=' scalar_exprs              { char *st = strdup($3.str); string out = plural_assignment_function($1, "scalars", st, "CollOfScalars", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' vector_exprs              { char *st = strdup($3.str); string out = plural_assignment_function($1, "vectors", st, "CollOfVectors", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' vertices                  { char *st = strdup($3.str); string out = plural_assignment_function($1, "vertices", st, "CollOfVertices", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' edges                     { char *st = strdup($3.str); string out = plural_assignment_function($1, "edges", st, "CollOfEdges", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' faces                     { char *st = strdup($3.str); string out = plural_assignment_function($1, "faces", st, "CollOfFaces", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' cells                     { char *st = strdup($3.str); string out = plural_assignment_function($1, "cells", st, "CollOfCells", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' adbs                      { char *st = strdup($3.str); string out = plural_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' boolean_exprs             { char *st = strdup($3.str); string out = plural_assignment_function($1, "bools", st, "CollOfBools", $3.size); $$ = strdup(out.c_str()); }
                 | VARIABLE '=' USCOS '(' plural ')'      { string out = USCOS_assignment_function($1, $5.str, $5.size); $$ = strdup(out.c_str()); }
                 ;


//if the variable hasn't been declared before, it is an assignment with deduced declaration (type)

assignment: singular_assignment     { char* out = strdup($1); $$ = out; }
          | plural_assignment       { char* out = strdup($1); $$ = out; }
          ;




singular_declaration_with_assignment: VARIABLE ':' SCALAR '=' scalar_expr          { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "scalar", st, "Scalar"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' VECTOR '=' vector_expr          { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "vector", st, "Vector"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' VERTEX '=' vertex               { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "vertex", st, "Vertex"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' EDGE '=' edge                   { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "edge", st, "Edge"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' FACE '=' face                   { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "face", st, "Face"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' CELL '=' cell                   { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "cell", st, "Cell"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' ADB '=' adb                     { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "scalarAD", st, "ScalarAD"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' BOOLEAN '=' boolean_expr        { char *st = strdup($5); string out = singular_declaration_with_assignment_function($1, "bool", st, "bool"); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' SCALAR '=' USS                  { string out = USS_declaration_with_assignment_function($1); $$ = strdup(out.c_str()); }
                                    | VARIABLE ':' SCALAR '=' USSWD '(' number ')' { string out = USSWD_declaration_with_assignment_function($1, $7); $$ = strdup(out.c_str()); }
                                    ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' scalar_exprs          { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "scalars", st, "CollOfScalars", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF VECTOR '=' vector_exprs          { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "vectors", st, "CollOfVectors", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF VERTEX '=' vertices              { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "vertices", st, "CollOfVertices", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF EDGE '=' edges                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "edges", st, "CollOfEdges", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF FACE '=' faces                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "faces", st, "CollOfFaces", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF CELL '=' cells                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "cells", st, "CollOfCells", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF ADB '=' adbs                     { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF BOOLEAN '=' boolean_exprs        { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1, "bools", st, "CollOfBools", $7.size); $$ = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF SCALAR '=' USCOS '(' plural ')'  { string out = USCOS_declaration_with_assignment_function($1, $9.str, $9.size); $$ = strdup(out.c_str()); }
                                  ;


extended_plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR ON plural '=' scalar_exprs          { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "scalars", st1, "CollOfScalars", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF VECTOR ON plural '=' vector_exprs          { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "vectors", st1, "CollOfVectors", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF VERTEX ON plural '=' vertices              { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "vertices", st1, "CollOfVertices", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF EDGE ON plural '=' edges                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "edges", st1, "CollOfEdges", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF FACE ON plural '=' faces                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "faces", st1, "CollOfFaces", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF CELL ON plural '=' cells                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "cells", st1, "CollOfCells", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF ADB ON plural '=' adbs                     { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "scalarsAD", st1, "CollOfScalarsAD", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural '=' boolean_exprs        { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1, "bools", st1, "CollOfBools", st2, $9.size, $7.size); $$ = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF SCALAR ON plural '=' USCOS '(' plural ')'  { string out = USCOS_extended_declaration_with_assignment_function($1, $11.str, $7.str, $11.size, $7.size); $$ = strdup(out.c_str()); }
                                           ;



 declaration_with_assignment: singular_declaration_with_assignment          { char* out = strdup($1); $$ = out; }
                            | plural_declaration_with_assignment            { char* out = strdup($1); $$ = out; }
                            | extended_plural_declaration_with_assignment   { char* out = strdup($1); $$ = out; }
                            ;




// instructions which can be used in the program and in a function's body
command: declaration                    { char* out = strdup($1); $$ = out; }
       | assignment                     { char* out = strdup($1); $$ = out; }
       | declaration_with_assignment    { char* out = strdup($1); $$ = out; }
       ;


command1: command                       { char* out = strdup($1); $$ = out; }
        | command COMMENT               { string st1 = $1; string st2 = $2; stringstream ss; ss << st1 << " // " << st2.substr(1, st2.size() - 1); $$ = strdup(ss.str().c_str()); }
        | COMMENT                       { string st1 = $1; stringstream ss; ss << "// " << st1.substr(1, st1.size() - 1); $$ = strdup(ss.str().c_str()); }
        ;


// instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                    { stringstream ss; ss << $1; $$ = strdup(ss.str().c_str()); }
        | function_declaration                       { stringstream ss; ss << $1; $$ = strdup(ss.str().c_str()); }
        | function_assignment                        { stringstream ss; ss << $1; $$ = strdup(ss.str().c_str()); }
        | output                                     { stringstream ss; ss << $1; $$ = strdup(ss.str().c_str()); }
    //  | function_declaration_with_assignment       { stringstream ss; ss << $1; $$ = strdup(ss.str().c_str()); }
        ;


pr: pr command2 '\n'                  {
                                        string out = $2;
                                        cout << out << endl;
                                        currentLineNumber++;
                                      }
  | pr command2 COMMENT '\n'          {
                                        string out1 = $2;
                                        string out2 = $3;
                                        cout << out1 << " // " << out2.substr(1, out2.size() - 1) << endl;   //+1 to skip comment sign (#)
                                        currentLineNumber++;
                                      }
  | pr COMMENT '\n'                   {
                                        string out = $2;
                                        cout << "// " << out.substr(1, out.size() - 1) << endl;      //+1 to skip comment sign (#)
                                        currentLineNumber++;
                                      }
  | pr '\n'                           { cout << endl; currentLineNumber++; }
  |                                   { }
  ;

%%


extern int yylex();
extern int yyparse();

int main()
{
  HEAP_CHECK();
  cout << "/*" << endl << "  Copyright 2013 SINTEF ICT, Applied Mathematics." << endl << "*/" << endl;
  cout << "#include <opm/core/utility/parameters/ParameterGroup.hpp>" << endl;
  cout << "#include <opm/core/linalg/LinearSolverFactory.hpp>" << endl;
  cout << "#include <opm/core/utility/ErrorMacros.hpp>" << endl;
  cout << "#include <opm/autodiff/AutoDiffBlock.hpp>" << endl;
  cout << "#include <opm/autodiff/AutoDiffHelpers.hpp>" << endl;
  cout << "#include <opm/core/grid.h>" << endl;
  cout << "#include <opm/core/grid/GridManager.hpp>" << endl;
  cout << "#include <algorithm>" << endl;
  cout << "#include <iterator>" << endl;
  cout << "#include <iostream>" << endl;
  cout << "#include <cmath>" << endl;
  cout << endl;
  cout << "#include \"EquelleRuntimeCPU.hpp\"" << endl;
  cout << endl << endl;
  cout << "int main()" << endl;
  cout << "{" << endl;
  cout << "Opm::parameter::ParameterGroup param(argc, argv, false);" << endl;
  cout << "EquelleRuntimeCPU er(param);" << endl;
  cout << "UserParameters up(param, er);" << endl;
  cout << endl;
  HEAP_CHECK();
  yyparse();
  cout << "}" << endl;
  HEAP_CHECK();
  return 0;
}


void yyerror(const char* s)
{
  HEAP_CHECK();
  string st = s;
  cout << st << endl;
}













// function which returns true if s2 is contained in s1
bool find1(char* s1, char* s2)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+()<>!=,");
  while(pch != NULL)
  {
      if(strcmp(pch, s2) == 0)
      {
          HEAP_CHECK();
          return true;
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return false;
}


// function which returns the first undeclared variable from a given expression (this function is called after the function "check1" returns false)
char* find2(char* s1)
{
  HEAP_CHECK();
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
                    if(strcmp(pch, var[i].name.c_str()) == 0)
                    {
                      found = true;
                      break;
                    }
                }
                if(found == false)
                {
                  HEAP_CHECK();
                  return pch;
                }
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return strdup("InvalidCall");
}


// function which returns the first unassigned variable from a given expression (this function is called after the function "check2" returns false)
char* find3(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < varNo; i++)
      {
          if(strcmp(pch, var[i].name.c_str()) == 0)
          {
              if(var[i].assigned == false)
              {
                HEAP_CHECK();
                return pch;
              }
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return strdup("InvalidCall");
}


// function which returns the number of parameters from a given parameters list
int find4(char *s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " ,");
  int counter = 0;
  while(pch != NULL)
  {
      counter++;
      pch = strtok (NULL, " ,");
  }
  HEAP_CHECK();
  return counter;
}


// function which returns the first undeclared variable from a given expression inside a function (this function is called after the function "check3" returns false)
char* find5(char* s1)
{
  HEAP_CHECK();
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
                    if(strcmp(pch, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0)
                    {
                      found = true;
                      break;
                    }

                if(found == false)
                    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                        if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
                        {
                          found = true;
                          break;
                        }

                if(found == false)
                {
                  HEAP_CHECK();
                  return pch;
                }
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return strdup("InvalidCall");
}


// function which returns the first unassigned variable from a given expression inside a function (this function is called after the function "check4" returns false)
char* find6(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
      {
          if(strcmp(pch, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0)
          {
              if(fun[currentFunctionIndex].localVariables[i].assigned == false)
              {
                HEAP_CHECK();
                return pch;
              }
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return strdup("InvalidCall");
}


// function which checks if each variable (one that begins with a small letter and it's not a default/user-defined function) from a given expression was declared
bool check1(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=^,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0 && strcmp(pch, "true") != 0 && strcmp(pch, "false") != 0 && strcmp(pch, "return") != 0)  // not a default function or a small letter keyword
        {
            if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
            {
                bool found = false;
                int i;
                for(i = 0; i < varNo; i++)
                {
                    if(strcmp(pch, var[i].name.c_str()) == 0)
                    {
                      found = true;
                      break;
                    }
                }
                if(found == false)
                {
                  HEAP_CHECK();
                  bool found2 = false;
                  int j;
                  for(j = 0; j < funNo; j++)
                  {
                      if(strcmp(pch, fun[j].name.c_str()) == 0)
                      {
                        found2 = true;
                        break;
                      }
                  }
                  if(found2 == false)   // the unfound name doesn't belong to a user-defined function either
                    return false;
                }
            }
        }
      }
      pch = strtok (NULL, " -+*/()<>!=^,");
  }
  HEAP_CHECK();
  return true;
}


// function which checks if each variable from a given expression was assigned to a value, and returns false if not
bool check2(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      for(i = 0; i < varNo; i++)
      {
          if(strcmp(pch, var[i].name.c_str()) == 0)
          {
              if(var[i].assigned == false)
              {
                HEAP_CHECK();
                return false;
              }
              break;
          }
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return true;
}


// function which checks if each variable from a given expression (which is inside a function) is declared as a header or local variable in the current function (indicated by a global index)
bool check3(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
      {
        if(strncmp(pch, "er.", 3) != 0)  // not a default function
        {
            if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
            {
                int i;
                bool taken = false;
                for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
                    if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), pch) == 0)
                    {
                        taken = true;
                        break;
                    }
                if(taken == false)
                    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                        if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
                        {
                            taken = true;
                            break;
                        }
                if(taken == false)
                {
                    HEAP_CHECK();
                    bool found = false;
                    int j;
                    for(j = 0; j < funNo; j++)
                    {
                        if(strcmp(pch, fun[j].name.c_str()) == 0)
                        {
                          found = true;
                          break;
                        }
                    }
                    if(found == false)    // the unfound name doesn't belong to a user-defined function either (user-defined functions can be called from inside another function's body)
                        return false;   // the given variable doesn't exist among the header and local variables of the current function
                }
            }
        }
      }

      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return true;    // all the variables from the given expression are declared inside the current function
}


// function which checks if each variable from a given expression (which is inside a function) is assigned as a header or local variable in the current function (indicated by a global index)
bool check4(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
      bool taken = false;
      for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
          if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), pch) == 0)
          {
              taken = true;     // if it's a header variable, it's already assigned
              break;
          }
      if(taken == false)
          for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
              if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
              {
                  if(fun[currentFunctionIndex].localVariables[i].assigned == false)
                  {
                      HEAP_CHECK();
                      return false;
                  }
                  break;
              }

      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return true;    // all the variables from the given expression are assigned inside the current function
}


// function which checks if the given variable corresponds to a header/local variable of the current function and if its type is the same as the current function's return type
bool check5(char* s1)
{
  HEAP_CHECK();
  bool found = false;
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
  {
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), s1) == 0)
    {
      found = true;
      break;
    }
  }
  if(found == true)
  {
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].type.c_str(), fun[currentFunctionIndex].returnType.c_str()) != 0 || (fun[currentFunctionIndex].headerVariables[i].length != fun[currentFunctionIndex].returnSize && fun[currentFunctionIndex].returnSize != ANY && fun[currentFunctionIndex].headerVariables[i].length != ANY))
    {
       HEAP_CHECK();
       return false;
    }
    HEAP_CHECK();
    return true;
  }

  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), s1) == 0)
    {
      found = true;
      break;
    }
  if(found == true)
  {
    if(strcmp(fun[currentFunctionIndex].localVariables[i].type.c_str(), fun[currentFunctionIndex].returnType.c_str()) != 0 || (fun[currentFunctionIndex].localVariables[i].length != fun[currentFunctionIndex].returnSize && fun[currentFunctionIndex].returnSize != ANY && fun[currentFunctionIndex].localVariables[i].length != ANY))
    {
      HEAP_CHECK();
      return false;
    }
    HEAP_CHECK();
    return true;
  }

  HEAP_CHECK();
  return false;
}


// function which checks if the phrase "length_mismatch_error" is found within a string (for error checking of length mismatch operations)
bool check6(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(strcmp(pch, "length_mismatch_error") == 0)
      {
          HEAP_CHECK();
          return true;
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return false;    // there is no length mismatch error contained in the given expression
}


// function which checks if the phrase "wrong_type_error" is found within a string (for error checking of operations between variables)
bool check7(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      if(strcmp(pch, "wrong_type_error") == 0)
      {
          HEAP_CHECK();
          return true;
      }
      pch = strtok (NULL, " -+*/()<>!=,");
  }
  HEAP_CHECK();
  return false;
}


// function which checks if a given array of variables corresponds to a given array of types
bool check8(char *s1, char *s2)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *cs2 = strdup(s2);    // we need to make a copy, because the strtok function modifies the given string
  char *pch1;
  pch1 = strtok(cs1, " ,");
  char *pch2;
  pch2 = strtok(cs2, " ,");
  while(pch1 != NULL && pch2 != NULL)   // they should terminate simultaneously
  {
      bool found = false;
      int i;
      for(i = 0; i < varNo; i++)
      {
          if(strcmp(pch1, var[i].name.c_str()) == 0)
          {
            found = true;
            break;
          }
      }
      if(found == false)
      {
        HEAP_CHECK();
        return false;
      }

      if(var[i].type != CPPToEquelle1(pch2))
      {
          return false;
      }

      pch1 = strtok (NULL, " ,");
      pch2 = strtok (NULL, " ,");
  }
  HEAP_CHECK();
  return true;
}


// function which checks if a given string contains any error message and, if so, it returns the appropriate message
string check9(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,:");
  bool errorFound = false;
  while(pch != NULL)
  {
      if(strcmp(pch, "error1") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error2") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error3") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error4") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error5") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error6") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      if(strcmp(pch, "error7") == 0)
      {
          HEAP_CHECK();
          errorFound = true;
          break;
      }
      pch = strtok (NULL, " -+*/()<>!=,:");
  }

  if(errorFound == true)
      return errorTypeToErrorMessage(pch);
  HEAP_CHECK();
  return "isOk";    // there are no error messages within the assignment (which may be caused by function calls)
}


// function which returns the type of a variable, based on its name
char* getType(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return strdup(var[i].type.c_str());
      }
  }
  HEAP_CHECK();
  return strdup("NoType");
}


// function which returns the index of a variable, based on its name
int getIndex1(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return i;
      }
  }
  HEAP_CHECK();
  return -1;
}


// function which returns the index of a function, based on its name
int getIndex2(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return i;
      }
  }
  HEAP_CHECK();
  return -1;
}


// function which returns the size of a variable, based on its name
double getSize1(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return var[i].length;
      }
  }
  HEAP_CHECK();
  return -1;
}


// function which returns the return size of a function, based on its name
double getSize2(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[i].returnSize;
      }
  }
  HEAP_CHECK();
  return -1;
}


// function which returns the size of a header/local variable inside the current function, based on its name
double getSize3(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[currentFunctionIndex].headerVariables[i].length;
      }
  }
  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[currentFunctionIndex].localVariables[i].length;
      }
  }
  HEAP_CHECK();
  return -1;
}


// function which counts the number of given arguments, separated by '@'
int getSize4(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " @");
  int ctr = 0;
  while(pch != NULL)
  {
      ctr++;
      pch = strtok (NULL, " @");
  }
  HEAP_CHECK();
  return ctr;
}


// function which receives the start of a function declaration and returns its name
char* extract(char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " =");
  HEAP_CHECK();
  return pch;
}


// function used to transfer a string within a structure to a separate memory address (of its own)
char *structureToString(char* st)
{
  HEAP_CHECK();
  char *strA = (char*)malloc(sizeof(char)*strlen(st)+1);
  strcpy(strA, st);
  HEAP_CHECK();
  return strA;
}


// function which converts a type from C++ to its corresponding type in Equelle
string CPPToEquelle1(char* st)
{
    if(strcmp(st, "Scalar") == 0) {
      return "scalar";
    }

    if(strcmp(st, "Vector") == 0) {
      return "vector";
    }

    if(strcmp(st, "Vertex") == 0) {
      return "vertex";
    }

    if(strcmp(st, "Edge") == 0) {
      return "edge";
    }

    if(strcmp(st, "Face") == 0) {
      return "face";
    }

    if(strcmp(st, "Cell") == 0) {
      return "cell";
    }

    if(strcmp(st, "ScalarAD") == 0) {
      return "scalarAD";
    }

    if(strcmp(st, "bool") == 0) {
      return "bool";
    }

    if(strcmp(st, "CollOfScalars") == 0) {
      return "scalars";
    }

    if(strcmp(st, "CollOfVectors") == 0) {
      return "vectors";
    }

    if(strcmp(st, "CollOfVertices") == 0) {
      return "vertices";
    }

    if(strcmp(st, "CollOfEdges") == 0) {
      return "edges";
    }

    if(strcmp(st, "CollOfCells") == 0) {
      return "cells";
    }

    if(strcmp(st, "CollOfScalarsAD") == 0) {
      return "scalarsAD";
    }

    if(strcmp(st, "CollOfBools") == 0) {
      return "bools";
    }

    return "InvalidType";
}


// function which returns the corresponding size of a C++ type
double CPPToEquelle2(char* st)
{
    if(strcmp(st, "Scalar") == 0) {
      return 1;
    }

    if(strcmp(st, "Vector") == 0) {
      return 1;
    }

    if(strcmp(st, "Vertex") == 0) {
      return 1;
    }

    if(strcmp(st, "Edge") == 0) {
      return 1;
    }

    if(strcmp(st, "Face") == 0) {
      return 1;
    }

    if(strcmp(st, "Cell") == 0) {
      return 1;
    }

    if(strcmp(st, "ScalarAD") == 0) {
      return 1;
    }

    if(strcmp(st, "bool") == 0) {
      return 1;
    }

    if(strcmp(st, "CollOfScalars") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfVectors") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfVertices") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfEdges") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfCells") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfScalarsAD") == 0) {
      return ANY;
    }

    if(strcmp(st, "CollOfBools") == 0) {
      return ANY;
    }

    return -1;
}


// function which converts a type from Equelle to its corresponding type in C++
char* EquelleToCPP(string st)
{
    if(st == "scalar") {
      return strdup("Scalar");
    }

    if(st == "vector") {
      return strdup("Vector");
    }

    if(st == "vertex") {
      return strdup("Vertex");
    }

    if(st == "edge") {
      return strdup("Edge");
    }

    if(st == "face") {
      return strdup("Face");
    }

    if(st == "cell") {
      return strdup("Cell");
    }

    if(st == "scalarAD") {
      return strdup("ScalarAD");
    }

    if(st == "bool") {
      return strdup("bool");
    }

    if(st == "scalars") {
      return strdup("CollOfScalars");
    }

    if(st == "vectors") {
      return strdup("CollOfVectors");
    }

    if(st == "vertices") {
      return strdup("CollOfVertices");
    }

    if(st == "edges") {
      return strdup("CollOfEdges");
    }

    if(st == "cells") {
      return strdup("CollOfCells");
    }

    if(st == "scalarsAD") {
      return strdup("CollOfScalarsAD");
    }

    if(st == "bools") {
      return strdup("CollOfBools");
    }

    return strdup("InvalidType");
}


// function used to convert possible error types received when assigning a function call to a variable and converts them to explicit messages
string errorTypeToErrorMessage(string errorType)
{
    // map<string, string> myMap;

    // myMap["error1"] = "One function from the assignment is not declared";
    // myMap["error2"] = "One function from the assignment is not assigned";
    // myMap["error3"] = "The return type of a function from the assignment has been declared with a different return type";
    // myMap["error4"] = "One function from the assignment receives a different number of arguments than its signature";
    // myMap["error5"] = "One function from the assignment receives an undefined variable as an argument";
    // myMap["error6"] = "One function from the assignment receives an unassigned variable as an argument";
    // myMap["error7"] = "One function from the assignment receives arguments which do not match the function's signature";

    // return myMap(errorType);

    if(errorType == "error1")
        return "One function from the assignment is not declared";
    if(errorType == "error2")
        return "One function from the assignment is not assigned";
    if(errorType == "error3")
        return "The return type of a function from the assignment has been declared with a different return type";
    if(errorType == "error4")
        return "One function from the assignment receives a different number of arguments than its signature";
    if(errorType == "error5")
        return "One function from the assignment receives an undefined variable as an argument";
    if(errorType == "error6")
        return "One function from the assignment receives an unassigned variable as an argument";
    if(errorType == "error7")
        return "One function from the assignment receives arguments which do not match the function's signature";
    return "InvalidCall";
}


string functionToAnySingularType(char *st1, char *st2, char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
      return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
      return "error2: The function is not assigned";
    }
    if(strcmp(EquelleToCPP(fun[getIndex2(st1)].returnType), st2) != 0)
    {
      stringstream ss;
      ss << "error3: The return type of the function is not a " << st4 << " type";
      return ss.str();
    }
    if(fun[getIndex2(st1)].noParam != getSize4(st3))
    {
      return "error4: The number of arguments of the function does not correspond to the number of arguments sent";
    }
    if(check1(st3) == false)
    {
      return "error5: One input variable from the function's call is undefined";
    }
    if(check2(st3) == false)
    {
      return "error6: One input variable from the function's call is unassigned";
    }
    if(check8(st3, strdup(fun[getIndex2(st1)].paramList.c_str())) == false)
    {
      return "error7: The parameter list of the template of the function does not correspond to the given parameter list";
    }

    stringstream ss;
    ss << st1 << "(" << st3 << ")";
    return ss.str();
}



string functionToAnyCollectionType(char *st1, char *st2, char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
      return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
      return "error2: The function is not assigned";
    }
    if(strcmp(EquelleToCPP(fun[getIndex2(st1)].returnType), st2) != 0)
    {
      stringstream ss;
      ss << "error3: The return type of the function is not a collection of " << st4 << " type";
      return ss.str();
    }
    if(fun[getIndex2(st1)].noParam != getSize4(st3))
    {
      return "error4: The number of arguments of the function does not correspond to the number of arguments sent";
    }
    if(check1(st3) == false)
    {
      return "error5: One input variable from the function's call is undefined";
    }
    if(check2(st3) == false)
    {
      return "error6: One input variable from the function's call is unassigned";
    }
    if(check8(st3, strdup(fun[getIndex2(st1)].paramList.c_str())) == false)
    {
      return "error7: The parameter list of the template of the function does not correspond to the given parameter list";
    }

    return "ok";
}



















string singular_declaration_function(char* st1, char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = 1;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            var[varNo++].name = st1;
            var[varNo-1].type = st2;
            var[varNo-1].length = 1;
            var[varNo-1].assigned = false;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string plural_declaration_function(char* st1, char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = ANY;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
              var[varNo++].name = st1;
              var[varNo-1].type = st2;
              var[varNo-1].length = ANY;
              var[varNo-1].assigned = false;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string extended_plural_declaration_function(char* st1, char* st2, char* st3, double d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                  if(check7(st3) == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the " << st2 << " variable '" << st1 << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check3(st3) == false)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is undeclared";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(check4(st3) == false)
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is unassigned";
                              finalString = ss.str();
                          }
                          else
                          {
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].length = d1;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
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
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
              if(check7(st3) == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the " << st2 << " variable '" << st1 << "'";
                  finalString = ss.str();
              }
              else
              {
                  if(check1(st3) == false)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is undeclared";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check2(st3) == false)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is unassigned";
                          finalString = ss.str();
                      }
                      else
                      {
                          var[varNo++].name = st1;
                          var[varNo-1].type = st2;
                          var[varNo-1].length = d1;
                          var[varNo-1].assigned = false;
                      }
                  }
              }
        }
    }

    HEAP_CHECK();
    return finalString;
}


string singular_assignment_function(char* st1, char* st2, char* st3, char* st4)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local " << st2 << " variable '" << st1 << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check9(st3) != "isOk")
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": " << check9(st3);
                          finalString = ss.str();
                      }
                      else
                      {
                          if(check6(st3) == true)
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(find1(st3, st1))
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check3(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check4(st3) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(st3) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              fun[currentFunctionIndex].localVariables[i].assigned = true;
                                              stringstream ss;
                                              ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                              finalString = ss.str();
                                          }
                                      }
                                  }
                              }
                          }
                      }
                  }
              else
              {   // deduced declaration
                  if(check9(st3) != "isOk")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": " << check9(st3);
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check6(st3) == true)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(find1(st3, st1))
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(check3(st3) == false)
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check4(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check7(st3) == true)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          stringstream ss;
                                          ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                          finalString = ss.str();
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
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
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st3) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st3);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st3, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(st3) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(st3) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(st3) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            var[i].assigned = true;
                                            stringstream ss;
                                            ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                            finalString = ss.str();
                                        }
                                    }
                                }
                            }
                        }
                    }
              }
        else
        {
            // deduced declaration
            if(check9(st3) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st3);
                finalString = ss.str();
            }
            else
            {
                if(check6(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st3, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st3) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = st2;
                                    var[varNo-1].length = 1;
                                    var[varNo-1].assigned = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    HEAP_CHECK();
    return finalString;
}


string plural_assignment_function(char* st1, char* st2, char* st3, char* st4, double d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local " << st2 << " variable '" << st1 << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                        if(check9(st3) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(st3);
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(st3) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(st3, st1))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(st3) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(st3) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(st3) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(getSize3(st1) != d1)
                                                    if(getSize3(st1) == ANY)
                                                    {
                                                        fun[currentFunctionIndex].localVariables[i].length = d1;
                                                        fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                        stringstream ss;
                                                        ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                        finalString = ss.str();
                                                    }
                                                    else
                                                    {
                                                        stringstream ss;
                                                        ss << "error at line " << currentLineNumber << ": The length of the " << st2 << " variable '" << st1 << "' from its definition differs than the length of its assignment in the function '" << fun[currentFunctionIndex].name << "'";
                                                        finalString = ss.str();
                                                    }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                    finalString = ss.str();
                                                    fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                }
                                            }
                                        }
                                    }
                                }
                          }
                      }
                  }
              else
              {   // deduced declaration
                  if(check9(st3) != "isOk")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": " << check9(st3);
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check6(st3) == true)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(find1(st3, st1))
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(check3(st3) == false)
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check4(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check7(st3) == true)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          stringstream ss;
                                          ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                          finalString = ss.str();
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
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
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st3) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st3);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st3, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(st3) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(st3) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(st3) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(getSize1(st1) != d1)
                                                if(getSize1(st1) == ANY)
                                                {
                                                    var[i].length = d1;
                                                    var[i].assigned = true;
                                                    stringstream ss;
                                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the " << st2 << " variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                finalString = ss.str();
                                                var[i].assigned = true;
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
            // deduced declaration
            if(check9(st3) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st3);
                finalString = ss.str();
            }
            else
            {
                if(check6(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st3, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st3) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = st2;
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

    HEAP_CHECK();
    return finalString;
}


string singular_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '%s'" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st3) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st3);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st3, st1))
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check3(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check4(st3) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(st3) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              stringstream ss;
                                              ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                              finalString = ss.str();
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
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
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(st3) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st3);
                finalString = ss.str();
            }
            else
            {
                if(check6(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st3, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st3) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = st2;
                                    var[varNo-1].length = 1;
                                    var[varNo-1].assigned = true;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    HEAP_CHECK();
    return finalString;
}


string plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, double d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '%s'" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st3) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st3);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st3, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                finalString = ss.str();
                            }
                              else
                              {
                                  if(check3(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check4(st3) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(st3) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              stringstream ss;
                                              ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                              finalString = ss.str();
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
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
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(st3) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st3);
                finalString = ss.str();
            }
            else
            {
                if(check6(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st3, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st3) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the " << st2 << " variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = st2;
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

    HEAP_CHECK();
    return finalString;
}


string extended_plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, char* st5, double d1, double d2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), st1) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), st1) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st3) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st3);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st3, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check3(st3) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check4(st3) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check3(st5) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find5(st5) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is undeclared";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check4(st5) == false)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": The variable '" << find6(st5) << "' contained in the ON expression of the " << st2 << " variable '" << st1 << "' is unassigned";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(check7(st5) == true)
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the " << st2 << " variable '" << st1 << "'";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    if(d2 != d1)
                                                    {
                                                        stringstream ss;
                                                        ss << "error at line " << currentLineNumber << ": The length of the " << st2 << " variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                        finalString = ss.str();
                                                    }
                                                    else
                                                    {
                                                        stringstream ss;
                                                        ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                        finalString = ss.str();
                                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = st1;
                                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = st2;
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
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(st3) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st3);
                finalString = ss.str();
            }
            else
            {
                if(check6(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st3, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                  if(check1(st5) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find2(st5) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check2(st5) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find3(st5) << "' contained in the definition of the " << st2 << " variable '" << st1 << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(st5) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the " << st2 << " variable '" << st1 << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              if(d2 != d1)
                                              {
                                                  stringstream ss;
                                                  ss << "error at line " << currentLineNumber << ": The length of the " << st2 << " variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                  finalString = ss.str();
                                              }
                                              else
                                              {
                                                  stringstream ss;
                                                  ss << "const " << st4 << " " << st1 << " = " << st3 << ";";
                                                  finalString = ss.str();
                                                  var[varNo++].name = st1;
                                                  var[varNo-1].type = st2;
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
    }

    HEAP_CHECK();
    return finalString;
}


















string USS_assignment_function(char* st1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                  if(var[i].type != "scalar")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is declared as a " << var[i].type << " and cannot be assigned to a scalar";
                      finalString = ss.str();
                  }
                  else
                  {
                      var[i].assigned = true;
                      stringstream ss;
                      ss << "const Scalar " << st1 << " = param.get<Scalar>(\"" << st1 << "\");";
                      finalString = ss.str();
                  }
              }
        else
        {
            // deduced declaration
            stringstream ss;
            ss << "const Scalar " << st1 << " = param.get<Scalar>(\"" << st1 << "\");";
            finalString = ss.str();
            var[varNo++].name = st1;
            var[varNo-1].type = "scalar";
            var[varNo-1].length = 1;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USS_declaration_with_assignment_function(char* st1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            stringstream ss;
            ss << "const Scalar " << st1 << " = param.get<Scalar>(\"" << st1 << "\");";
            finalString = ss.str();
            var[varNo++].name = st1;
            var[varNo-1].type = "scalar";
            var[varNo-1].length = 1;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_assignment_function(char* st1, char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar with default inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                  if(var[i].type != "scalar")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is declared as a " << var[i].type << " and cannot be assigned to a scalar";
                      finalString = ss.str();
                  }
                  else
                  {
                      var[i].assigned = true;
                      stringstream ss;
                      ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
                      finalString = ss.str();
                  }
              }
        else
        {
            // deduced declaration
            stringstream ss;
            ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
            finalString = ss.str();
            var[varNo++].name = st1;
            var[varNo-1].type = "scalar";
            var[varNo-1].length = 1;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_declaration_with_assignment_function(char* st1, char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            stringstream ss;
            ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
            finalString = ss.str();
            var[varNo++].name = st1;
            var[varNo-1].type = "scalar";
            var[varNo-1].length = 1;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_assignment_function(char* st1, char* st2, double d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(st2) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st2);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st2) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st2, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(st2) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(st2) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(st2) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(getSize1(st1) != d1)
                                                if(getSize1(st1) == ANY)
                                                {
                                                    var[i].length = d1;
                                                    var[i].assigned = true;
                                                    stringstream ss;
                                                    ss << "const CollOfScalars " << st1 << " = param.get<CollOfScalars>(\"" << st1 << "\", " << st2 << ");";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const CollOfScalars " << st1 << " = param.get<CollOfScalars>(\"" << st1 << "\", " << st2 << ");";
                                                finalString = ss.str();
                                                var[i].assigned = true;
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
            // deduced declaration
            if(check9(st2) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st2);
                finalString = ss.str();
            }
            else
            {
                if(check6(st2) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st2, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st2) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st2) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const CollOfScalars " << st1 << " = param.get<CollOfScalars>(\"" << st1 << "\", " << st2 << ");";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = "scalars";
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

    HEAP_CHECK();
    return finalString;
}


string USCOS_declaration_with_assignment_function(char* st1, char* st2, double d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(st2) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st2);
                finalString = ss.str();
            }
            else
            {
                if(check6(st2) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st2, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st2) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(st2) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const CollOfScalars " << st1 << " = param.get<CollOfScalars>(\"" << st1 << "\", " << st2 << ");";
                                    finalString = ss.str();
                                    var[varNo++].name = st1;
                                    var[varNo-1].type = "scalars";
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

    HEAP_CHECK();
    return finalString;
}


string USCOS_extended_declaration_with_assignment_function(char* st1, char* st2, char* st3, double d1, double d2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(st2) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(st2);
                finalString = ss.str();
            }
            else
            {
                if(check6(st2) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(st2, st1))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(st2) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                  if(check1(st3) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check2(st3) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(st3) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << st1 << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              if(d2 != d1)
                                              {
                                                  stringstream ss;
                                                  ss << "error at line " << currentLineNumber << ": The length of the variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                  finalString = ss.str();
                                              }
                                              else
                                              {
                                                  stringstream ss;
                                                  ss << "const CollOfScalars " << st1 << " = param.get<CollOfScalars>(\"" << st1 << "\", " << st3 << ");";
                                                  finalString = ss.str();
                                                  var[varNo++].name = st1;
                                                  var[varNo-1].type = "scalars";
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
    }

    HEAP_CHECK();
    return finalString;
}


string output_function(char* st1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be outputted inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore != true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' needs to be declared first";
            finalString = ss.str();
        }
        else
        {
            if(var[i].assigned != true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' needs to be assigned first";
                finalString = ss.str();
            }
            else
            {
                stringstream ss;
                ss << "er.output(\"" << st1 << "\", " << st1 << ");";
                finalString = ss.str();
            }
        }
    }

    HEAP_CHECK();
    return finalString;
}
