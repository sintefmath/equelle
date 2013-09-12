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
#include <stdbool.h>
#include <sstream>
#include <string>
#include <string.h>
#ifndef _MSC_VER
  #define HEAP_CHECK()
#else
  #include <crtdbg.h>
  #define HEAP_CHECK() _ASSERTE( _CrtCheckMemory( ) )
#endif

using namespace std;

void yyerror(const char* s);
int yylex(void);
bool find1(string s1, string s2);
string find2(string s1);
string find3(string s1);
int find4(string s1);
string find5(string s1);
string find6(string s1);
bool check1(string s1);
bool check2(string s1);
bool check3(string s1);
bool check4(string s1);
bool check5(string s1);
bool check6(string s1);
bool check7(string s1);
bool check8(string s1, string s2);
string getType(string s1);
int getIndex1(string s1);
int getIndex2(string s1);
double getSize1(string s1);
double getSize2(string s1);
double getSize3(string s1);
int getSize4(string s1);
string extract(string s1);
string CPPToEquelle1(string st);
int CPPToEquelle2(string st);
string singular_declaration_function(string st1, string st2);
string plural_declaration_function(string st1, string st2);
string extended_plural_declaration_function(string st1, string st2, string st3, double d1);
string singular_assignment_function(string st1, string st2, string st3, string st4);
string plural_assignment_function(string st1, string st2, string st3, string st4, double d1);
string singular_declaration_with_assignment_function(string st1, string st2, string st3, string st4);
string plural_declaration_with_assignment_function(string st1, string st2, string st3, string st4, double d1);
string extended_plural_declaration_with_assignment_function(string st1, string st2, string st3, string st4, string st5, double d1, double d2);





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
  string paramList;                           // (Cell, Face, Collection Of Vectors, Collection Of Scalars On AllFaces(Grid))
  VariableStructure headerVariables[100];     // (c1, f1, pv1, ps1)
  int noParam;                                // 4
  VariableStructure localVariables[100];      // var1, var2, var3
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
      string str;
  };

  // struct array
  // {
  //     char *sizes;
  //     char *str;
  // };

  struct dinfo
  {
      string cCode;
      string sepCode;
  };
}



%union
{
  int value;
  string str;           // the non-terminals which need to store only the translation code for C++ will be declared with this type
  struct info inf;      // the non-terminals which need to store both the translation code for C++ and the size of the collection will be declared with this type
  //struct array arr;   // the values which are passed as arguments when calling a function must be checked to see if they correspond to the function's template
  struct dinfo dinf;
};


%%


scalar_expr: scalar_term 		                 {$$ = $1;}
           | '-' scalar_term                 {stringstream ss; ss << "-" << $2; $$ = ss.str();}
           | scalar_expr '+' scalar_term	   {stringstream ss; ss << $1 << " + " << $3; $$ = ss.str();}
           | scalar_expr '-' scalar_term	   {stringstream ss; ss << $1 << " - " << $3; $$ = ss.str();}
           ;


scalar_term: scalar_factor		                       {$$ = $1;}
           | scalar_term '*' scalar_factor	         {stringstream ss; ss << $1 << " * " << $3; $$ = ss.str();}
           | scalar_factor '*' scalar_term           {stringstream ss; ss << $1 << " * " << $3; $$ = ss.str();}
           | scalar_term '/' scalar_factor	         {stringstream ss; ss << $1 << " / " << $3; $$ = ss.str();}
           | scalar_term '^' scalar_factor           {stringstream ss; ss << "er.pow(" << $1 << ", " << $3 << ")"; $$ = ss.str();}
           ;


scalar_factor: NUMBER		                               {$$ = $1;}
             | NUMBER '.' NUMBER                       {stringstream ss; ss << $1 << "." << $3; $$ = ss.str();}
             | '(' scalar_expr ')'	                   {stringstream ss; ss << "(" << $2 << ")"; $$ = ss.str();}
             | EUCLIDEAN_LENGTH '(' vector_expr ')'    {stringstream ss; ss << "er.euclideanLength(" << $3 << ")"; $$ = ss.str();}
             | LENGTH '(' edge ')'                     {stringstream ss; ss << "er.length(" << $3 << ")"; $$ = ss.str();}
             | AREA '(' face ')'                       {stringstream ss; ss << "er.area(" << $3 << ")"; $$ = ss.str();}
             | VOLUME '(' cell ')'                     {stringstream ss; ss << "er.volume(" << $3 << ")"; $$ = ss.str();}
             | DOT '(' vector_expr ',' vector_expr ')' {stringstream ss; ss << "er.dot(" << $3 << ", " << $5 << ")"; $$ = ss.str();}
             | CEIL '(' scalar_expr ')'                {stringstream ss; ss << "er.ceil(" << $3 << ")"; $$ = ss.str();}
             | FLOOR '(' scalar_expr ')'               {stringstream ss; ss << "er.floor(" << $3 << ")"; $$ = ss.str();}
             | ABS '(' scalar_expr ')'                 {stringstream ss; ss << "er.abs(" << $3 << ")"; $$ = ss.str();}
             | MIN '(' scalars ')'                     {stringstream ss; ss << "er.min(" << $3 << ")"; $$ = ss.str();}
             | MAX '(' scalars ')'                     {stringstream ss; ss << "er.max(" << $3 << ")"; $$ = ss.str();}
             | VARIABLE                                {
                                                          if(strcmp(getType($1), "scalar") != 0)
                                                          {
                                                              stringstream ss;
                                                              ss << "wrong_type_error  " << $1;   // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                              $$ = ss.str();
                                                          }
                                                          else
                                                          {
                                                              $$ = $1;
                                                          }
                                                       }

             | VARIABLE '(' values ')'                 {
                                                          if(getIndex2($1) == -1)
                                                          {
                                                            stringstream ss;
                                                            ss << "error1: This function does not exist";
                                                            $$ = ss.str();
                                                          }
                                                          else
                                                              if(fun[getIndex2($1)].assigned == false)
                                                              {
                                                                stringstream ss;
                                                                ss << "error2: The function is not assigned";
                                                                $$ = ss.str();
                                                              }
                                                              else
                                                                  if(strcmp(fun[getIndex2($1)].returnType, "Scalar") != 0)
                                                                  {
                                                                    stringstream ss;
                                                                    ss << "error3: The return type of the function is not a scalar type";
                                                                    $$ = ss.str();
                                                                  }
                                                                  else
                                                                      if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                      {
                                                                        stringstream ss;
                                                                        ss << "error4: The number of arguments of the function does not correspond to the number of arguments sent";
                                                                        $$ = ss.str();
                                                                      }
                                                                      else
                                                                          if(check1($3) == false)
                                                                          {
                                                                            stringstream ss;
                                                                            ss << "error5: One input variable from the function's call is undefined";
                                                                            $$ = ss.str();
                                                                          }
                                                                          else
                                                                              if(check2($3) == false)
                                                                              {
                                                                                stringstream ss;
                                                                                ss << "error6: One input variable from the function's call is unassigned";
                                                                                $$ = ss.str();
                                                                              }
                                                                              else
                                                                                  if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                  {
                                                                                    stringstream ss;
                                                                                    ss << "error7: The parameter list of the template of the function does not correspond to the given parameter list";
                                                                                    $$ = ss.str();
                                                                                  }
                                                                                  else
                                                                                  {
                                                                                      stringstream ss;
                                                                                      ss << $1 << "(" << $3 << ")";
                                                                                      $$ = ss.str();
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
            | scalar_terms '^' scalar_factor    {char *str = append4("er.pow", '(', $1.str, ',', $3, ')'); $$.str = strdup(str); free(str); $$.size = $1.size;}
            ;


scalar_factors: EUCLIDEAN_LENGTH '(' vector_exprs ')'           {char *str = append3("er.euclideanLength", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | LENGTH '(' edges ')'                            {char *str = append3("er.length", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | AREA '(' faces ')'                              {char *str = append3("er.area", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | VOLUME '(' cells ')'                            {char *str = append3("er.volume", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | DOT '(' vector_exprs ',' vector_exprs ')'
                                                                {
                                                                   if($3.size != $5.size)    // check that the lengths of the 2 terms are equal
                                                                       $$.str = strdup("length_mismatch_error");
                                                                   else
                                                                   {
                                                                       char *str = append4("er.dot", '(', $3.str, ',', $5.str, ')');
                                                                       $$.str = strdup(str);
                                                                       free(str);
                                                                       $$.size = $3.size;
                                                                   }
                                                                }

              | CEIL '(' scalar_exprs ')'                       {char *str = append3("er.ceil", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | FLOOR '(' scalar_exprs ')'                      {char *str = append3("er.floor", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
              | ABS '(' scalar_exprs ')'                        {char *str = append3("er.abs", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
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

              | VARIABLE '(' values ')'             {
                                                        if(getIndex2($1) == -1)
                                                            sprintf($$.str, "error1: This function does not exist");
                                                        else
                                                            if(fun[getIndex2($1)].assigned == false)
                                                                sprintf($$.str, "error2: The function is not assigned");
                                                            else
                                                                if(strcmp(fun[getIndex2($1)].returnType, "CollOfScalars") != 0)
                                                                    sprintf($$.str, "error3: The return type of the function is not a collection of scalars type");
                                                                else
                                                                    if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                        sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                    else
                                                                        if(check1($3) == false)
                                                                            sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                        else
                                                                            if(check2($3) == false)
                                                                                sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                            else
                                                                                if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                    sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                                else
                                                                                {
                                                                                    $$.str = append3($1, '(', $3, ')');
                                                                                    $$.size = fun[getIndex2($1)].returnSize;
                                                                                }
                                                    }
              ;


numbers: NUMBER                         {$$ = $1;}
       | numbers ',' NUMBER             {stringstream ss; ss << $1 << ", " << $3; $$ = ss.str();}
       ;


vector_expr: vector_term                      {$$ = $1;}
           | '-' vector_term                  {stringstream ss; ss << "-" << $2; $$ = ss.str();}
           | vector_expr '+' vector_term      {stringstream ss; ss << $1 << " + " << $3; $$ = ss.str();}
           | vector_expr '-' vector_term      {stringstream ss; ss << $1 << " - " << $3; $$ = ss.str();}
           ;


vector_term: '(' numbers ')'                       {stringstream ss; ss << "(" << $2 << ")"; $$ = ss.str();}
           | CENTROID '(' cell ')'                 {stringstream ss; ss << "er.centroid(" << $3 << ")"; $$ = ss.str();}
           | NORMAL '(' face ')'                   {stringstream ss; ss << "er.normal(" << $3 << ")"; $$ = ss.str();}
           | '(' vector_expr ')'                   {stringstream ss; ss << "(" << $2 << ")"; $$ = ss.str();}              // produces 1 shift/reduce conflict
           | vector_term '*' scalar_factor         {stringstream ss; ss << $1 << " * " << $3; $$ = ss.str();}             // produces 1 reduce/reduce conflict
           | scalar_factor '*' vector_term         {stringstream ss; ss << $1 << " * " << $3; $$ = ss.str();}
           | vector_term '/' scalar_factor         {stringstream ss; ss << $1 << " / " << $3; $$ = ss.str();}
           | VARIABLE                              {
                                                      if(strcmp(getType($1), "vector") != 0)
                                                      {
                                                          stringstream ss;
                                                          ss << "wrong_type_error  " << $1;     // we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
                                                          $$ = ss.str();
                                                      }
                                                      else
                                                      {
                                                          $$ = $1;
                                                      }
                                                   }

           | VARIABLE '(' values ')'               {
                                                      if(getIndex2($1) == -1)
                                                          sprintf($$, "error1: This function does not exist");
                                                      else
                                                          if(fun[getIndex2($1)].assigned == false)
                                                              sprintf($$, "error2: The function is not assigned");
                                                          else
                                                              if(strcmp(fun[getIndex2($1)].returnType, "Vector") != 0)
                                                                  sprintf($$, "error3: The return type of the function is not a vector type");
                                                              else
                                                                  if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                      sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                  else
                                                                      if(check1($3) == false)
                                                                          sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                      else
                                                                          if(check2($3) == false)
                                                                              sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                          else
                                                                              if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                  sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                              else
                                                                              {
                                                                                  $$ = append3($1, '(', $3, ')');
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
            | CENTROID '(' cells ')'                 {char *str = append3("er.centroid", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
            | NORMAL '(' faces ')'                   {char *str = append3("er.normal", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
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

            | VARIABLE '(' values ')'                {
                                                        if(getIndex2($1) == -1)
                                                            sprintf($$.str, "error1: This function does not exist");
                                                        else
                                                            if(fun[getIndex2($1)].assigned == false)
                                                                sprintf($$.str, "error2: The function is not assigned");
                                                            else
                                                                if(strcmp(fun[getIndex2($1)].returnType, "CollOfVectors") != 0)
                                                                    sprintf($$.str, "error3: The return type of the function is not a collection of vectors type");
                                                                else
                                                                    if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                        sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                    else
                                                                        if(check1($3) == false)
                                                                            sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                        else
                                                                            if(check2($3) == false)
                                                                                sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                            else
                                                                                if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                    sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                                else
                                                                                {
                                                                                    $$.str = append3($1, '(', $3, ')');
                                                                                    $$.size = fun[getIndex2($1)].returnSize;
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

      | VARIABLE '(' values ')'               {
                                                  if(getIndex2($1) == -1)
                                                      sprintf($$, "error1: This function does not exist");
                                                  else
                                                      if(fun[getIndex2($1)].assigned == false)
                                                          sprintf($$, "error2: The function is not assigned");
                                                      else
                                                          if(strcmp(fun[getIndex2($1)].returnType, "Vertex") != 0)
                                                              sprintf($$, "error3: The return type of the function is not a vertex type");
                                                          else
                                                              if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                  sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                              else
                                                                  if(check1($3) == false)
                                                                      sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                  else
                                                                      if(check2($3) == false)
                                                                          sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                      else
                                                                          if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                              sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                          else
                                                                          {
                                                                              $$ = append3($1, '(', $3, ')');
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

        | VARIABLE '(' values ')'                 {
                                                      if(getIndex2($1) == -1)
                                                          sprintf($$.str, "error1: This function does not exist");
                                                      else
                                                          if(fun[getIndex2($1)].assigned == false)
                                                              sprintf($$.str, "error2: The function is not assigned");
                                                          else
                                                              if(strcmp(fun[getIndex2($1)].returnType, "CollOfVertices") != 0)
                                                                  sprintf($$.str, "error3: The return type of the function is not a collection of vertices type");
                                                              else
                                                                  if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                      sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                  else
                                                                      if(check1($3) == false)
                                                                          sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                      else
                                                                          if(check2($3) == false)
                                                                              sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                          else
                                                                              if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                  sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                              else
                                                                              {
                                                                                  $$.str = append3($1, '(', $3, ')');
                                                                                  $$.size = fun[getIndex2($1)].returnSize;
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

    | VARIABLE '(' values ')'               {
                                                if(getIndex2($1) == -1)
                                                    sprintf($$, "error1: This function does not exist");
                                                else
                                                    if(fun[getIndex2($1)].assigned == false)
                                                        sprintf($$, "error2: The function is not assigned");
                                                    else
                                                        if(strcmp(fun[getIndex2($1)].returnType, "Edge") != 0)
                                                            sprintf($$, "error3: The return type of the function is not an edge type");
                                                        else
                                                            if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                            else
                                                                if(check1($3) == false)
                                                                    sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                else
                                                                    if(check2($3) == false)
                                                                        sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                    else
                                                                        if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                            sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                        else
                                                                        {
                                                                            $$ = append3($1, '(', $3, ')');
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

     | VARIABLE '(' values ')'                  {
                                                    if(getIndex2($1) == -1)
                                                        sprintf($$.str, "error1: This function does not exist");
                                                    else
                                                        if(fun[getIndex2($1)].assigned == false)
                                                            sprintf($$.str, "error2: The function is not assigned");
                                                        else
                                                            if(strcmp(fun[getIndex2($1)].returnType, "CollOfEdges") != 0)
                                                                sprintf($$.str, "error3: The return type of the function is not a collection of edges type");
                                                            else
                                                                if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                    sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                else
                                                                    if(check1($3) == false)
                                                                        sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                    else
                                                                        if(check2($3) == false)
                                                                            sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                        else
                                                                            if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                            else
                                                                            {
                                                                                $$.str = append3($1, '(', $3, ')');
                                                                                $$.size = fun[getIndex2($1)].returnSize;
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

    | VARIABLE '(' values ')'               {
                                                if(getIndex2($1) == -1)
                                                    sprintf($$, "error1: This function does not exist");
                                                else
                                                    if(fun[getIndex2($1)].assigned == false)
                                                        sprintf($$, "error2: The function is not assigned");
                                                    else
                                                        if(strcmp(fun[getIndex2($1)].returnType, "Face") != 0)
                                                            sprintf($$, "error3: The return type of the function is not a face type");
                                                        else
                                                            if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                            else
                                                                if(check1($3) == false)
                                                                    sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                else
                                                                    if(check2($3) == false)
                                                                        sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                    else
                                                                        if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                            sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                        else
                                                                        {
                                                                            $$ = append3($1, '(', $3, ')');
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

     | VARIABLE '(' values ')'                   {
                                                    if(getIndex2($1) == -1)
                                                        sprintf($$.str, "error1: This function does not exist");
                                                    else
                                                        if(fun[getIndex2($1)].assigned == false)
                                                            sprintf($$.str, "error2: The function is not assigned");
                                                        else
                                                            if(strcmp(fun[getIndex2($1)].returnType, "CollOfFaces") != 0)
                                                                sprintf($$.str, "error3: The return type of the function is not a collection of faces type");
                                                            else
                                                                if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                    sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                else
                                                                    if(check1($3) == false)
                                                                        sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                    else
                                                                        if(check2($3) == false)
                                                                            sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                        else
                                                                            if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                            else
                                                                            {
                                                                                $$.str = append3($1, '(', $3, ')');
                                                                                $$.size = fun[getIndex2($1)].returnSize;
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

    | VARIABLE '(' values ')'                 {
                                                  if(getIndex2($1) == -1)
                                                      sprintf($$, "error1: This function does not exist");
                                                  else
                                                      if(fun[getIndex2($1)].assigned == false)
                                                          sprintf($$, "error2: The function is not assigned");
                                                      else
                                                          if(strcmp(fun[getIndex2($1)].returnType, "Cell") != 0)
                                                              sprintf($$, "error3: The return type of the function is not a cell type");
                                                          else
                                                              if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                  sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                              else
                                                                  if(check1($3) == false)
                                                                      sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                  else
                                                                      if(check2($3) == false)
                                                                          sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                      else
                                                                          if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                              sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                          else
                                                                          {
                                                                              $$ = append3($1, '(', $3, ')');
                                                                          }
                                              }
    ;


cells: INTERIOR_CELLS '(' GRID ')'          {$$.str = strdup("er.interiorCells()"); $$.size = INTERIORCELLS;}
     | BOUNDARY_CELLS '(' GRID ')'          {$$.str = strdup("er.boundaryCells()"); $$.size = BOUNDARYCELLS;}
     | ALL_CELLS '(' GRID ')'               {$$.str = strdup("er.allCells()"); $$.size = ALLCELLS;}
     | FIRST_CELL '(' faces ')'             {char *str = append3("er.firstCell", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
     | SECOND_CELL '(' faces ')'            {char *str = append3("er.secondCell", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
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

     | VARIABLE '(' values ')'                  {
                                                    if(getIndex2($1) == -1)
                                                        sprintf($$.str, "error1: This function does not exist");
                                                    else
                                                        if(fun[getIndex2($1)].assigned == false)
                                                            sprintf($$.str, "error2: The function is not assigned");
                                                        else
                                                            if(strcmp(fun[getIndex2($1)].returnType, "CollOfCells") != 0)
                                                                sprintf($$.str, "error3: The return type of the function is not a collection of cells type");
                                                            else
                                                                if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                    sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                else
                                                                    if(check1($3) == false)
                                                                        sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                    else
                                                                        if(check2($3) == false)
                                                                            sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                        else
                                                                            if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                            else
                                                                            {
                                                                                $$.str = append3($1, '(', $3, ')');
                                                                                $$.size = fun[getIndex2($1)].returnSize;
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

   | VARIABLE '(' values ')'                {
                                                if(getIndex2($1) == -1)
                                                    sprintf($$, "error1: This function does not exist");
                                                else
                                                    if(fun[getIndex2($1)].assigned == false)
                                                        sprintf($$, "error2: The function is not assigned");
                                                    else
                                                        if(strcmp(fun[getIndex2($1)].returnType, "ScalarAD") != 0)
                                                            sprintf($$, "error3: The return type of the function is not a scalarAD type");
                                                        else
                                                            if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                            else
                                                                if(check1($3) == false)
                                                                    sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                else
                                                                    if(check2($3) == false)
                                                                        sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                    else
                                                                        if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                            sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                        else
                                                                        {
                                                                            $$ = append3($1, '(', $3, ')');
                                                                        }
                                            }
   ;


adbs: GRADIENT '(' adbs ')'       {char *str = append3("er.negGradient", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
    | DIVERGENCE '(' adbs ')'     {char *str = append3("er.divergence", '(', $3.str, ')'); $$.str = strdup(str); free(str); $$.size = $3.size;}
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

    | VARIABLE '(' values ')'                   {
                                                    if(getIndex2($1) == -1)
                                                        sprintf($$.str, "error1: This function does not exist");
                                                    else
                                                        if(fun[getIndex2($1)].assigned == false)
                                                            sprintf($$.str, "error2: The function is not assigned");
                                                        else
                                                            if(strcmp(fun[getIndex2($1)].returnType, "CollOfScalarsAD") != 0)
                                                                sprintf($$.str, "error3: The return type of the function is not a collection of scalarsAD type");
                                                            else
                                                                if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                    sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                else
                                                                    if(check1($3) == false)
                                                                        sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                    else
                                                                        if(check2($3) == false)
                                                                            sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                        else
                                                                            if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                            else
                                                                            {
                                                                                $$.str = append3($1, '(', $3, ')');
                                                                                $$.size = fun[getIndex2($1)].returnSize;
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

            | VARIABLE '(' values ')'               {
                                                        if(getIndex2($1) == -1)
                                                            sprintf($$, "error1: This function does not exist");
                                                        else
                                                            if(fun[getIndex2($1)].assigned == false)
                                                                sprintf($$, "error2: The function is not assigned");
                                                            else
                                                                if(strcmp(fun[getIndex2($1)].returnType, "bool") != 0)
                                                                    sprintf($$, "error3: The return type of the function is not a bool type");
                                                                else
                                                                    if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                        sprintf($$, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                    else
                                                                        if(check1($3) == false)
                                                                            sprintf($$, "error5: One input variable from the function's call is undefined");
                                                                        else
                                                                            if(check2($3) == false)
                                                                                sprintf($$, "error6: One input variable from the function's call is unassigned");
                                                                            else
                                                                                if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                    sprintf($$, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                                else
                                                                                {
                                                                                    $$ = append3($1, '(', $3, ')');
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

             | VARIABLE '(' values ')'                {
                                                          if(getIndex2($1) == -1)
                                                              sprintf($$.str, "error1: This function does not exist");
                                                          else
                                                              if(fun[getIndex2($1)].assigned == false)
                                                                  sprintf($$.str, "error2: The function is not assigned");
                                                              else
                                                                  if(strcmp(fun[getIndex2($1)].returnType, "CollOfBools") != 0)
                                                                      sprintf($$.str, "error3: The return type of the function is not a collection of bools type");
                                                                  else
                                                                      if(fun[getIndex2($1)].noParam != getSize4($3))
                                                                          sprintf($$.str, "error4: The number of arguments of the function does not correspond to the number of arguments sent");
                                                                      else
                                                                          if(check1($3) == false)
                                                                              sprintf($$.str, "error5: One input variable from the function's call is undefined");
                                                                          else
                                                                              if(check2($3) == false)
                                                                                  sprintf($$.str, "error6: One input variable from the function's call is unassigned");
                                                                              else
                                                                                  if(check8($3, fun[getIndex2($1)].paramList) == false)
                                                                                      sprintf($$.str, "error7: The parameter list of the template of the function does not correspond to the given parameter list");
                                                                                  else
                                                                                  {
                                                                                      $$.str = append3($1, '(', $3, ')');
                                                                                      $$.size = fun[getIndex2($1)].returnSize;
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


header: VARIABLE HEADER_DECL SCALAR                          {char *str = append9("Scalar", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL VECTOR                          {char *str = append9("Vector", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL VERTEX                          {char *str = append9("Vertex", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL EDGE                            {char *str = append9("Edge", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL FACE                            {char *str = append9("Face", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL CELL                            {char *str = append9("Cell", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL ADB                             {char *str = append9("ScalarAD", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL BOOLEAN                         {char *str = append9("bool", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF SCALAR            {char *str = append9("CollOfScalars", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF VECTOR            {char *str = append9("CollOfVectors", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF VERTEX            {char *str = append9("CollOfVertices", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF EDGE              {char *str = append9("CollOfEdges", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF FACE              {char *str = append9("CollOfFaces", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF CELL              {char *str = append9("CollOfCells", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF ADB               {char *str = append9("CollOfScalarsAD", ' ', $1); $$ = strdup(str); free(str);}
      | VARIABLE HEADER_DECL COLLECTION OF BOOLEAN           {char *str = append9("CollOfBools", ' ', $1); $$ = strdup(str); free(str);}
      ;


parameter_list: header                         {$$ = strdup($1);}
              | parameter_list ',' header      {char *str = append5($1,',',$3); $$ = strdup(str); free(str);}
              ;


commands: command                              {$$ = strdup($1);}
        | commands end_lines command           {char *str = append6($1, $2, $3); $$ = strdup(str); free(str);}
        |                                      {$$ = strdup("");}     // a function can have only the return instruction
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
values: VARIABLE                {$$ = strdup($1);}
      | values ',' VARIABLE     {char *str = append5($1,',',$3); $$ = strdup(str); free(str);}
      ;


end_lines: '\n'                 {char *s = (char *)malloc(sizeof(char)); s[0] = '\n'; s[1] = '\0'; $$ = strdup(s); free(s);}
         | '\n' end_lines       {char *str = append7('\n', $2); $$ = strdup(str); free(str);}
         |                      {$$ = strdup("");}
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
                        sprintf($$.str, "return %s ? %s : %s", $2, $4, $6);
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
                        sprintf($$.str, "return %s", $2);
                        $$.size = getSize3($2);
                    }
                  }
            ;


function_start: VARIABLE '=' end_lines '{'
                                            {
                                              insideFunction = true;
                                              currentFunctionIndex = getIndex2($1);
                                              char *str = append12($1, '=', $3, '{');
                                              $$ = strdup(str);
                                              free(str);
                                            }


// these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: VARIABLE ':' FUNCTION '(' parameter_list ')' RET type
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
                                                {
                                                    stringstream ss;
                                                    ss << "error: The function " << $1 << "' is redeclared";
                                                    $$ = ss.str();
                                                }
                                                else
                                                {
                                                        fun[funNo++].name = strdup($1);
                                                        fun[funNo-1].returnType = strdup($8.str);
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
                                                          char *aux1 = structureToString(fun[funNo-1].paramList);
                                                          fun[funNo-1].paramList = append1(aux1, ',', pch2);
                                                          char *copy = strdup(pch2);
                                                          pch2 = strtok(NULL, " ");   // name of the variable

                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam++].name = strdup(pch2);
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].type = CPPToEquelle1(copy);    // the string we have as a parameter list is already transformed in C++, but we need the types' keywords from Equelle
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].length = CPPToEquelle2(copy);  // the string we have as a parameter list is already transformed in C++, but we need the types' lengths
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].assigned = true;

                                                          pch = strtok(NULL, ",");
                                                        }

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
                                                      {
                                                          stringstream ss;
                                                          ss << "error: The function '" << fun[i].name << "' is reassigned";
                                                          $$ = ss.str();
                                                      }
                                                      else
                                                      {
                                                          if($5.size != -1)
                                                          {
                                                              char *str = (char *)malloc(1000 * sizeof(char));
                                                              str = append13($1, $2, $3, $4, $5.str, $6, '}');
                                                              $$ = strdup(str);
                                                              free(str);
                                                              if(fun[i].returnSize == ANY && $5.size != ANY)
                                                                  fun[i].returnSize = $5.size;
                                                              else
                                                                  if(fun[i].returnSize != ANY && $5.size == ANY)
                                                                      ;   // do nothing (the function must keep its return size from the definition)
                                                                  else
                                                                      ;   // if both are ANY, the function's return type is already correct; if none are ANY, then they should already be equal, otherwise the instruction flow wouldn't enter on this branch
                                                              fun[i].assigned = true;
                                                          }
                                                          else
                                                          {
                                                              stringstream ss;
                                                              ss << "error: At least one of the return variables does not exist within the function or the return type of the function '" << fun[i].name << "' from its assignment differs than the length of the return type from the function's definition";
                                                              $$ = ss.str();
                                                          }

                                                      }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error: The function '" << extract($1) <<"' must be declared before being assigned";
                                                    $$ = ss.str();
                                                }
                                                insideFunction = false;
                                                currentFunctionIndex = -1;
                                            }
                   ;




// function_declaration_with_assignment: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type '=' end_lines '{' end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
//                                     ; // tre sa punem booleana globala true inainte sa execute comenzile din functie





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


extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON plural      {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "scalars", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF VECTOR ON plural      {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "vectors", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF VERTEX ON plural      {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "vertices", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF EDGE ON plural        {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "edges", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF FACE ON plural        {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "faces", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF CELL ON plural        {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "cells", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF ADB ON plural         {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "scalarsAD", st, $7.size); $$ = out;}
                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural     {char *st = structureToString($7.str); char* out = extended_plural_declaration_function($1, "bools", st, $7.size); $$ = out;}
                           ;


declaration: singular_declaration           {$$ = strdup($1);}
           | plural_declaration             {$$ = strdup($1);}
           | extended_plural_declaration    {$$ = strdup($1);}
           ;



singular_assignment: VARIABLE '=' scalar_expr              {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "scalar", str, "Scalar"); $$ = out;}
                   | VARIABLE '=' vector_expr              {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "vector", str, "Vector"); $$ = out;}
                   | VARIABLE '=' vertex                   {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "vertex", str, "Vertex"); $$ = out;}
                   | VARIABLE '=' edge                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "edge", str, "Edge"); $$ = out;}
                   | VARIABLE '=' face                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "face", str, "Face"); $$ = out;}
                   | VARIABLE '=' cell                     {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "cell", str, "Cell"); $$ = out;}
                   | VARIABLE '=' adb                      {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "scalarAD", str, "ScalarAD"); $$ = out;}
                   | VARIABLE '=' boolean_expr             {char str[1000]; int i; for(i = 0; i < strlen($3); i++) str[i] = $3[i]; str[strlen($3)] = '\0'; char* out = singular_assignment_function($1, "bool", str, "bool"); $$ = out;}
                   ;


plural_assignment: VARIABLE '=' scalar_exprs              {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "scalars", st, "CollOfScalars", $3.size); $$ = out;}
                 | VARIABLE '=' vector_exprs              {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "vectors", st, "CollOfVectors", $3.size); $$ = out;}
                 | VARIABLE '=' vertices                  {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "vertices", st, "CollOfVertices", $3.size); $$ = out;}
                 | VARIABLE '=' edges                     {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "edges", st, "CollOfEdges", $3.size); $$ = out;}
                 | VARIABLE '=' faces                     {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "faces", st, "CollOfFaces", $3.size); $$ = out;}
                 | VARIABLE '=' cells                     {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "cells", st, "CollOfCells", $3.size); $$ = out;}
                 | VARIABLE '=' adbs                      {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $3.size); $$ = out;}
                 | VARIABLE '=' boolean_exprs             {char *st = structureToString($3.str); char* out = plural_assignment_function($1, "bools", st, "CollOfBools", $3.size); $$ = out;}
                 ;


//if the variable hasn't been declared before, it is an assignment with deduced declaration (type)

assignment: singular_assignment     {char* out = strdup($1); $$ = out;}
          | plural_assignment       {char* out = strdup($1); $$ = out;}
          ;




singular_declaration_with_assignment: VARIABLE ':' SCALAR '=' scalar_expr          {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "scalar", str, "Scalar"); $$ = out;}
                                    | VARIABLE ':' VECTOR '=' vector_expr          {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "vector", str, "Vector"); $$ = out;}
                                    | VARIABLE ':' VERTEX '=' vertex               {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "vertex", str, "Vertex"); $$ = out;}
                                    | VARIABLE ':' EDGE '=' edge                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "edge", str, "Edge"); $$ = out;}
                                    | VARIABLE ':' FACE '=' face                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "face", str, "Face"); $$ = out;}
                                    | VARIABLE ':' CELL '=' cell                   {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "cell", str, "Cell"); $$ = out;}
                                    | VARIABLE ':' ADB '=' adb                     {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "scalarAD", str, "ScalarAD"); $$ = out;}
                                    | VARIABLE ':' BOOLEAN '=' boolean_expr        {char str[1000]; int i; for(i = 0; i < strlen($5); i++) str[i] = $5[i]; str[strlen($5)] = '\0'; char* out = singular_declaration_with_assignment_function($1, "bool", str, "bool"); $$ = out;}
                                    ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' scalar_exprs        {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "scalars", st, "CollOfScalars", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF VECTOR '=' vector_exprs        {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "vectors", st, "CollOfVectors", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF VERTEX '=' vertices            {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "vertices", st, "CollOfVertices", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF EDGE '=' edges                 {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "edges", st, "CollOfEdges", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF FACE '=' faces                 {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "faces", st, "CollOfFaces", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF CELL '=' cells                 {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "cells", st, "CollOfCells", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF ADB '=' adbs                   {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", $7.size); $$ = out;}
                                  | VARIABLE ':' COLLECTION OF BOOLEAN '=' boolean_exprs      {char *st = structureToString($7.str); char* out = plural_declaration_with_assignment_function($1, "bools", st, "CollOfBools", $7.size); $$ = out;}
                                  ;


extended_plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR ON plural '=' scalar_exprs        {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "scalars", st, "CollOfScalars", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF VECTOR ON plural '=' vector_exprs        {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "vectors", st, "CollOfVectors", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF VERTEX ON plural '=' vertices            {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "vertices", st, "CollOfVertices", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF EDGE ON plural '=' edges                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "edges", st, "CollOfEdges", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF FACE ON plural '=' faces                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "faces", st, "CollOfFaces", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF CELL ON plural '=' cells                 {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "cells", st, "CollOfCells", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF ADB ON plural '=' adbs                   {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "scalarsAD", st, "CollOfScalarsAD", st2, $9.size, $7.size); $$ = out;}
                                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural '=' boolean_exprs      {char *st = structureToString($9.str); char *st2 = structureToString($7.str); char* out = extended_plural_declaration_with_assignment_function($1, "bools", st, "CollOfBools", st2, $9.size, $7.size); $$ = out;}
                                           ;


 declaration_with_assignment: singular_declaration_with_assignment          {char* out = strdup($1); $$ = out;}
                            | plural_declaration_with_assignment            {char* out = strdup($1); $$ = out;}
                            | extended_plural_declaration_with_assignment   {char* out = strdup($1); $$ = out;}
                            ;




// instructions which can be used in the program and in a function's body
command: declaration                    {char* out = strdup($1); $$ = out;}
       | assignment                     {char* out = strdup($1); $$ = out;}
       | declaration_with_assignment    {char* out = strdup($1); $$ = out;}
       ;


// instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                    {char* out = strdup($1); $$ = out;}
        | function_declaration                       {char* out = strcat($1,";"); $$ = out;}
        | function_assignment                        {char* out = strdup($1); $$ = out;}
    //  | function_declaration_with_assignment       {$$ = strdup($1);}
        ;


pr: pr command2 '\n'                  {
                                        char* out = $2;
                                        printf("%s\n", out);
                                        currentLineNumber++;
                                      }
  | pr command2 COMMENT '\n'          {
                                        char* out1 = $2;
                                        char* out2 = $3;
                                        printf("%s // %s \n", out1, (out2+1)); //+1 to skip comment sign (#)
                                        currentLineNumber++;
                                      }
  | pr COMMENT '\n'                   {
	                                      char* out = $2;
	                                      printf("// %s\n", (out+1)); //+1 to skip comment sign (#)
	                                      currentLineNumber++;
                                      }
  | pr '\n'                           {printf("\n"); currentLineNumber++;}
  |                                   {}
  ;

%%


extern int yylex();
extern int yyparse();

int main()
{
  HEAP_CHECK();
  cout << "Opm::parameter::ParameterGroup param(argc, argv, false);" << endl << "EquelleRuntimeCPU er(param);" << endl << "UserParameters up(param, er);" << endl << endl;
  HEAP_CHECK();
  yyparse();
  HEAP_CHECK();
  return 0;
}


void yyerror(const char* s)
{
  HEAP_CHECK();
  cout << s;
}




bool find1(string s1, string s2)     // function which returns true if s2 is contained in s1
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


string find2(string s1)   // function which returns the first undeclared variable from a given expression (this function is called after the function "check1" returns false)
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
                    if(strcmp(pch, var[i].name) == 0)
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


string find3(string s1)     // function which returns the first unassigned variable from a given expression (this function is called after the function "check2" returns false)
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
          if(strcmp(pch, var[i].name) == 0)
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


int find4(char *s1)       // function which returns the number of parameters from a given parameters list
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


string find5(string s1)   // function which returns the first undeclared variable from a given expression inside a function (this function is called after the function "check3" returns false)
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


string find6(string s1)     // function which returns the first unassigned variable from a given expression inside a function (this function is called after the function "check4" returns false)
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
          if(strcmp(pch, fun[currentFunctionIndex].localVariables[i].name) == 0)
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


bool check1(string s1)   // function which checks if each variable (one that begins with a small letter and it's not a function) from a given expression was declared
{
  HEAP_CHECK();
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
                {
				          HEAP_CHECK();
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


bool check2(string s1)     // function which checks if each variable from a given expression was assigned to a value, and returns false if not
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
          if(strcmp(pch, var[i].name) == 0)
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


bool check3(string s1)     // function which checks if each variable from a given expression (which is inside a function) is declared as a header or local variable in the current function (indicated by a global index)
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
                {
					          HEAP_CHECK();
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


bool check4(string s1)     // function which checks if each variable from a given expression (which is inside a function) is assigned as a header or local variable in the current function (indicated by a global index)
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


bool check5(string s1)     // function which checks if the given variable corresponds to a header/local variable of the current function and if its type is the same as the current function's return type
{
  HEAP_CHECK();
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
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].type, fun[currentFunctionIndex].returnType) != 0 || (fun[currentFunctionIndex].headerVariables[i].length != fun[currentFunctionIndex].returnSize && fun[currentFunctionIndex].returnSize != ANY && fun[currentFunctionIndex].headerVariables[i].length != ANY))
    {
	     HEAP_CHECK();
       return false;
	  }
	  HEAP_CHECK();
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
    if(strcmp(fun[currentFunctionIndex].localVariables[i].type, fun[currentFunctionIndex].returnType) != 0 || (fun[currentFunctionIndex].localVariables[i].length != fun[currentFunctionIndex].returnSize && fun[currentFunctionIndex].returnSize != ANY && fun[currentFunctionIndex].localVariables[i].length != ANY))
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


bool check6(string s1)     // function which checks if the phrase "length_mismatch_error" is found within a string (for error checking of length mismatch operations)
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


bool check7(string s1)    // function which checks if the phrase "wrong_type_error" is found within a string (for error checking of operations between variables)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " -+*/()<>!=,");
  while(pch != NULL)
  {
      int i;
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


bool check8(char *s1, char *s2)    // function which checks if a given array of variables corresponds to a given array of types
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
          if(strcmp(pch1, var[i].name) == 0)
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

      if(strcmp(var[i].type, pch2) != 0)
          return false;

      pch1 = strtok (NULL, " ,");
      pch2 = strtok (NULL, " ,");
  }
  HEAP_CHECK();
  return true;
}


string getType(string s1)     // function which returns the type of a variable, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
      {
		    HEAP_CHECK();
        return var[i].type;
	    }
  }
  HEAP_CHECK();
  return strdup("NoType");
}


int getIndex1(string s1)     // function which returns the index of a variable, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
      {
		    HEAP_CHECK();
        return i;
	    }
  }
  HEAP_CHECK();
  return -1;
}


int getIndex2(string s1)     // function which returns the index of a function, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name) == 0)
      {
		    HEAP_CHECK();
        return i;
	    }
  }
  HEAP_CHECK();
  return -1;
}


double getSize1(string s1)     // function which returns the size of a variable, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name) == 0)
      {
		    HEAP_CHECK();
        return var[i].length;
	    }
  }
  HEAP_CHECK();
  return -1;
}


double getSize2(string s1)     // function which returns the return size of a function, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name) == 0)
      {
		    HEAP_CHECK();
        return fun[i].returnSize;
	    }
  }
  HEAP_CHECK();
  return -1;
}


double getSize3(string s1)     // function which returns the size of a header/local variable inside the current function, based on its name
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].headerVariables[i].name) == 0)
      {
		    HEAP_CHECK();
        return fun[currentFunctionIndex].headerVariables[i].length;
	    }
  }
  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].localVariables[i].name) == 0)
      {
		    HEAP_CHECK();
        return fun[currentFunctionIndex].localVariables[i].length;
	    }
  }
  HEAP_CHECK();
  return -1;
}


int getSize4(string s1)    // function which counts the number of given arguments, separated by '@'
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


char* extract(string s1)   // function which receives the start of a function declaration and returns its name
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " =");
  HEAP_CHECK();
  return pch;
}


char *structureToString(string st)     // function used to transfer a string within a structure to a separate memory address (of its own)
{
  HEAP_CHECK();
  int a = strlen(st);
  char *strA = (char*)malloc(sizeof(char)*strlen(st)+1);
	strcpy(strA, st);
  HEAP_CHECK();
  return strA;
}


string CPPToEquelle1(string st)      // function which converts a type from C++ to its corresponding type in Equelle
{
    if(strcmp(st, "Scalar") == 0) {
      return strdup("scalar");
    }

    if(strcmp(st, "Vector") == 0) {
      return strdup("vector");
    }

    if(strcmp(st, "Vertex") == 0) {
      return strdup("vertex");
    }

    if(strcmp(st, "Edge") == 0) {
      return strdup("edge");
    }

    if(strcmp(st, "Face") == 0) {
      return strdup("face");
    }

    if(strcmp(st, "Cell") == 0) {
      return strdup("cell");
    }

    if(strcmp(st, "ScalarAD") == 0) {
      return strdup("scalarAD");
    }

    if(strcmp(st, "bool") == 0) {
      return strdup("bool");
    }

    if(strcmp(st, "CollOfScalars") == 0) {
      return strdup("scalars");
    }

    if(strcmp(st, "CollOfVectors") == 0) {
      return strdup("vectors");
    }

    if(strcmp(st, "CollOfVertices") == 0) {
      return strdup("vertices");
    }

    if(strcmp(st, "CollOfEdges") == 0) {
      return strdup("edges");
    }

    if(strcmp(st, "CollOfCells") == 0) {
      return strdup("cells");
    }

    if(strcmp(st, "CollOfScalarsAD") == 0) {
      return strdup("scalarsAD");
    }

    if(strcmp(st, "CollOfBools") == 0) {
      return strdup("bools");
    }

    return strdup("InvalidType");
}


int CPPToEquelle2(string st)      // function which returns the corresponding size of a C++ type
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
















string singular_declaration_function(string st1, string st2)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
            if(strcmp(var[i].name, st1) == 0)
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


string plural_declaration_function(string st1, string st2)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
            if(strcmp(var[i].name, st1) == 0)
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


string extended_plural_declaration_function(string st1, string st2, string st3, double d1)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
            if(strcmp(var[i].name, st1) == 0)
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


string singular_assignment_function(string st1, string st2, string st3, string st4)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
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
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local " << st2 << " variable '" << st1 << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
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
                                          ss << "const " << st4 << " " << st1 << " = " << st3;
                                          finalString = ss.str();
                                      }
                                  }
                              }
                          }
                      }
                  }
              else
              {   // deduced declaration
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
                                      ss << "const " << st4 << " " << st1 << " = " << st3;
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
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is reassigned";
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
                                        ss << "const " << st4 << " " << st1 << " = " << st3;
                                        finalString = ss.str();
                                    }
                                }
                            }
                        }
                    }
              }
        else
        {
            // deduced declaration
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
                                ss << "const " << st4 << " " << st1 << " = " << st3;
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

    HEAP_CHECK();
    return finalString;
}


string plural_assignment_function(string st1, string st2, string st3, string st4, double d1)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
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
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local " << st2 << " variable '" << st1 << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
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
                                                  ss << "const " << st4 << " " << st1 << " = " << st3;
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
                                              ss << "const " << st4 << " " << st1 << " = " << st3;
                                              finalString = ss.str();
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
                                      ss << "const " << st4 << " " << st1 << " = " << st3;
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
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is reassigned";
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
                                                ss << "const " << st4 << " " << st1 << " = " << st3;
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
                                            ss << "const " << st4 << " " << st1 << " = " << st3;
                                            finalString = ss.str();
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
            // deduced declaration
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
                                ss << "const " << st4 << " " << st1 << " = " << st3;
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

    HEAP_CHECK();
    return finalString;
}


string singular_declaration_with_assignment_function(string st1, string st2, string st3, string st4)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
                                          ss << "const " << st4 << " " << st1 << " = " << st3;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
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
                                ss << "const " << st4 << " " << st1 << " = " << st3;
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

    HEAP_CHECK();
    return finalString;
}


string plural_declaration_with_assignment_function(string st1, string st2, string st3, string st4, double d1)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
                                          ss << "const " << st4 << " " << st1 << " = " << st3;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
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
                                ss << "const " << st4 << " " << st1 << " = " << st3;
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

    HEAP_CHECK();
    return finalString;
}


string extended_plural_declaration_with_assignment_function(string st1, string st2, string st3, string st4, string st5, double d1, double d2)
{
    HEAP_CHECK();
    string finalString;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name, st1) == 0)
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
                                                    ss << "const " << st4 << " " << st1 << " = " << st3;
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
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The " << st2 << " variable '" << st1 << "' is redeclared";
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
                                              ss << "const " << st4 << " " << st1 << " = " << st3;
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

    HEAP_CHECK();
    return finalString;
}
