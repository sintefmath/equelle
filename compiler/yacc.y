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


/**
  * Definitions we need for types etc.
  */
%code top
{
	#include <iostream>
	#include <cstdio>
	#include <sstream>
	#include <string>
	#include <list>
	#include <cstring>



	// MACROS
	#ifndef _MSC_VER
	  #define HEAP_CHECK()
	#else
	  #pragma warning( disable : 4127 )
	  #include <crtdbg.h>
	  #define HEAP_CHECK() _ASSERTE(_CrtCheckMemory())
	#endif

	#define STREAM_TO_DOLLARS_CHAR_ARRAY(dd, streamcontent)                 do { stringstream ss; ss << streamcontent; dd = strdup(ss.str().c_str()); } while (false)
	#define LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY(dd)           do { stringstream ss; ss << "length_mismatch_error"; dd = strdup(ss.str().c_str()); } while (false)
	// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
	#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)                          do { stringstream ss; ss << "wrong_type_error  " << d1; dd = strdup(ss.str().c_str()); }  while (false)
	// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
	#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)            do { stringstream ss; ss << "wrong_type_error  " << d1; dd = strdup(ss.str().c_str()); }  while (false)

	//using std::cout;
	//using std::endl;
	//using std::string;
	using namespace std;

} //Code top


%code requires
{
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

	enum VariableType {
		TYPE_SCALAR,
		TYPE_VECTOR,
		TYPE_VERTEX,
		TYPE_EDGE,
		TYPE_FACE,
		TYPE_CELL,
		TYPE_BOOLEAN,
		TYPE_INVALID,
	};

	struct info
	{
		//Why does this not work?: info() : size(GRID_MAPPING_INVALID), array_size(-1), str(NULL), type(TYPE_INVALID), collection(false) {}
    char* str;            // The string which is to be outputted in the C++ file
		GridMapping grid_mapping;  // This defines the mapping of the variable (one value per face, cell, interior face, etc.)
		int array_size;       // The number of elements in a vector / array
		VariableType type;    // The type of the variable
		bool collection;      // Is this variable a collection of types or not?
	};







	// global structure and counter for storing the names of the variables of each type (used for stopping variables reassignment)
	struct VariableStructureForCPP
	{
	  string name;           // must begin with a small letter
	  string type;           // can be: scalar, vector, vertex, scalars etc.
	  GridMapping grid_mapping; // if the type is a singular type, then the length is 1; otherwise it can be any other number >= 1
	  bool assigned;         // we want to know if a variable has been assigned, in order to prevent errors (example: operations with unassigned variables)
	};


	// global structure and counter for storing the names of the functions
	struct FunctionStructureForCPP
	{
	  string name;                                      // g1
	  string returnType;                                // Collection Of Scalars
	  GridMapping grid_mapping;                        // GRID_MAPPING_ALLCELLS
	  string paramList;                                 // (Cell, Face, CollOfVectors, CollOfScalars On AllFaces(Grid))
	  VariableStructureForCPP headerVariables[100];     // (c1, f1, pv1, ps1)
	  int noParam;                                      // 4
	  VariableStructureForCPP localVariables[100];      // var1, var2, var3
	  string signature;                                 // (Cell c1, Face f1, CollOfVectors pv1, CollOfScalars On AllFaces(Grid) ps1)
	  int noLocalVariables;                             // 3
	  bool assigned;                                    // false
	};

} //Code requires








%code provides
{
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
	GridMapping getSize1(char* s1);
	GridMapping getSize2(char* s1);
	GridMapping getSize3(char* s1);
	int getSize4(char* s1);
	char* extract(char* s1);
	string CPPToEquelle1(char* st);
	GridMapping CPPToEquelle2(char* st);
	char* EquelleToCPP(string st);
	string errorTypeToErrorMessage(string errorType);
	string functionToAnySingularType(char *st1, char *st2, char *st3, const string &st4);
	string functionToAnyCollectionType(char *st1, char *st2, char *st3, const string &st4);
	string singular_declaration_function(char* st1, char* st2);
	string plural_declaration_function(char* st1, char* st2);
	string extended_plural_declaration_function(char* st1, char* st2, char* st3, GridMapping d1);
	string singular_assignment_function(char* st1, char* st2, char* st3, char* st4);
	string plural_assignment_function(char* st1, char* st2, char* st3, char* st4, GridMapping d1);
	string singular_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4);
	string plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, GridMapping d1);
	string extended_plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, char* st5, GridMapping d1, GridMapping d2);
	string USS_assignment_function(char* st1);
	string USS_declaration_with_assignment_function(char* st1);
	string USSWD_assignment_function(char* st1, char* st2);
	string USSWD_declaration_with_assignment_function(char* st1, char* st2);
	string USCOS_assignment_function(char* st1, char* st2, GridMapping d1);
	string USCOS_declaration_with_assignment_function(char* st1, char* st2, GridMapping d1);
	string USCOS_extended_declaration_with_assignment_function(char* st1, char* st2, char* st3, GridMapping d1, GridMapping d2);
	string output_function(char* st1);
	string getVariableTypeString1(VariableType v, bool collection);
  string getVariableTypeString2(VariableType v, bool collection);
} //Code provides


/**
  * All global variables
  * Should preferably not really be needed
  */
%code
{

VariableStructureForCPP var[10000];
FunctionStructureForCPP fun[10000];

int varNo = 0;
int funNo = 0;

bool insideFunction = false;
int currentFunctionIndex = -1;

int currentLineNumber = 1;

} //Code


%type<inf> floating_point
%type<inf> number
%type<inf> scalar_expr
%type<inf> scalar_exprs
%type<inf> scalar_term
%type<inf> scalar_terms
%type<inf> scalar_factor
%type<inf> scalar_factors
%type<inf> scalars
%type<inf> vector_expr
%type<inf> vector_term
%type<inf> vectors
%type<inf> vector_exprs
%type<inf> vector_terms
%type<inf> vertex
%type<inf> vertices
%type<inf> edge
%type<inf> edges
%type<inf> face
%type<inf> faces
%type<inf> cell
%type<inf> cells
%type<inf> adb
%type<inf> adbs
%type<inf> boolean_expr
%type<inf> boolean_term
%type<inf> boolean_exprs
%type<inf> boolean_terms
%type<inf> INTEGER
%type<inf> VARIABLE
%type<inf> COMMENT
%type<inf> plural
%type<inf> header
%type<inf> parameter_list
%type<inf> type
%type<inf> values
%type<inf> end_lines
%type<inf> return_instr
%type<inf> function_start
%type<inf> function_declaration
%type<inf> function_assignment
%type<inf> commands
%type<inf> command
%type<inf> command1;
%type<inf> command2
%type<inf> singular_declaration
%type<inf> plural_declaration
%type<inf> extended_plural_declaration
%type<inf> singular_assignment
%type<inf> plural_assignment
%type<inf> singular_declaration_with_assignment
%type<inf> plural_declaration_with_assignment
%type<inf> extended_plural_declaration_with_assignment
%type<inf> declaration
%type<inf> assignment
%type<inf> declaration_with_assignment
%type<inf> output
%type<inf> expression
%type<inf> expressions









%union
{
  struct info inf;
};


%%


floating_point: INTEGER '.' INTEGER          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "." << $3.str); $$.grid_mapping = GRID_MAPPING_ENTITY; $$.array_size = 1; $$.type = TYPE_SCALAR; $$.collection = false; }
              ;


number: INTEGER                              { $$.str = strdup($1.str); $$.grid_mapping = GRID_MAPPING_ENTITY; $$.array_size = 1; $$.type = TYPE_SCALAR; $$.collection = false; }
      | floating_point                       { $$.str = strdup($1.str); $$.grid_mapping = GRID_MAPPING_ENTITY; $$.array_size = 1; $$.type = TYPE_SCALAR; $$.collection = false; }
      ;


scalar_expr: scalar_term                     { $$.str = strdup($1.str); $$.grid_mapping = GRID_MAPPING_ENTITY; $$.array_size = 1; $$.type = TYPE_SCALAR; $$.collection = false; }
           | '-' scalar_term                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); }
           | scalar_expr '+' scalar_term     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str); }
           | scalar_expr '-' scalar_term     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str); }
           ;


scalar_term: scalar_factor                           { $$.str = strdup($1.str); }
           | scalar_term '*' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str); }
           | scalar_factor '*' scalar_term           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str); }
           | scalar_term '/' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " / " << $3.str); }
           | scalar_term '^' scalar_factor           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.pow(" << $1.str << ", " << $3.str << ")"); }
           ;


scalar_factor: number                                  { $$.str = strdup($1.str); }
             | '(' scalar_expr ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); }
             | EUCLIDEAN_LENGTH '(' vector_expr ')'    { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.euclideanLength(" << $3.str << ")"); }
             | LENGTH '(' edge ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.length(" << $3.str << ")"); }
             | AREA '(' face ')'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.area(" << $3.str << ")"); }
             | VOLUME '(' cell ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.volume(" << $3.str << ")"); }
             | DOT '(' vector_expr ',' vector_expr ')' { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.dot(" << $3.str << ", " << $5.str << ")"); }
             | CEIL '(' scalar_expr ')'                { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.ceil(" << $3.str << ")"); }
             | FLOOR '(' scalar_expr ')'               { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.floor(" << $3.str << ")"); }
             | ABS '(' scalar_expr ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.abs(" << $3.str << ")"); }
             | MIN '(' scalars ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.min(" << $3.str << ")"); }
             | MAX '(' scalars ')'                     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.max(" << $3.str << ")"); }
             | GRADIENT '(' scalar_expr ')'            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); }
             | DIVERGENCE '(' scalar_expr ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); }
             | VARIABLE                                {
                                                          if(strcmp(getType($1.str), "scalar") != 0)
                                                          {
                                                              WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup($1.str);
                                                          }
                                                       }

             | VARIABLE '(' values ')'                 {
                                                          string st = functionToAnySingularType($1.str, "Scalar", $3.str, "scalar");
                                                          $$.str = strdup(st.c_str());
                                                       }
             ;


scalars: scalar_exprs                 { $$.str = strdup($1.str); $$.array_size = $1.array_size; }
       | scalars ',' scalar_exprs     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); $$.array_size = $1.array_size + $3.array_size; }
       | scalar_expr                  { $$.str = strdup($1.str); $$.array_size = 1; }
       | scalars ',' scalar_expr      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); $$.array_size = $1.array_size + 1; }
       ;


scalar_exprs: scalar_terms                     { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
            | '-' scalar_terms                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); $$.grid_mapping = $2.grid_mapping; }
            | scalar_exprs '+' scalar_terms
                                               {
                                                  if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str);
                                                      $$.grid_mapping = $1.grid_mapping;
                                                  }
                                               }

            | scalar_exprs '-' scalar_terms
                                               {
                                                  if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str);
                                                      $$.grid_mapping = $1.grid_mapping;
                                                  }
                                               }
            ;


scalar_terms: scalar_factors                    { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
            | scalar_terms '*' scalar_factors
                                                {
                                                  if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str);
                                                      $$.grid_mapping = $1.grid_mapping;
                                                  }
                                               }
            | scalar_factors '*' scalar_terms
                                                {
                                                  if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str);
                                                      $$.grid_mapping = $1.grid_mapping;
                                                  }
                                               }
            | scalar_terms '/' scalar_factors
                                               {
                                                  if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                  {
                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                  }
                                                  else
                                                  {
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " / " << $3.str);
                                                      $$.grid_mapping = $1.grid_mapping;
                                                  }
                                               }
            | scalar_terms '^' scalar_factor   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.pow(" << $1.str << ", " << $3.str << ")"); $$.grid_mapping = $1.grid_mapping; }
            ;


scalar_factors: EUCLIDEAN_LENGTH '(' vector_exprs ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.euclideanLength(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
              | LENGTH '(' edges ')'                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.length(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
              | AREA '(' faces ')'                              { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.area(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
              | VOLUME '(' cells ')'                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.volume(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
              | DOT '(' vector_exprs ',' vector_exprs ')'
                                                                {
                                                                   if($3.grid_mapping != $5.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                                   {
                                                                      LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                                  }
                                                                   else
                                                                   {
                                                                       STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.dot(" << $3.str << ", " << $5.str << ")");
                                                                       $$.grid_mapping = $3.grid_mapping;
                                                                   }
                                                                }

              | CEIL '(' scalar_exprs ')'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.ceil(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
              | FLOOR '(' scalar_exprs ')'                      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.floor(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
              | ABS '(' scalar_exprs ')'                        { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.abs(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
              | GRADIENT '(' scalar_exprs ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
              | DIVERGENCE '(' scalar_exprs ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
              | VARIABLE                                        {
                                                                    if(strcmp(getType($1.str), "scalars") != 0)
                                                                    {
                                                                        WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                                    }
                                                                    else
                                                                    {
                                                                        $$.str = strdup($1.str);
                                                                        $$.grid_mapping = getSize1($1.str);
                                                                    }
                                                                }

              | VARIABLE '(' values ')'                         {
                                                                    string st = functionToAnyCollectionType($1.str, "CollOfScalars", $3.str, "scalars");
                                                                    if(st == "ok")
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                                        $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                                    }
                                                                    else
                                                                    {
                                                                        $$.str = strdup(st.c_str());
                                                                    }
                                                                }
              ;


vector_expr: vector_term                      { $$.str = strdup($1.str); }
           | '-' vector_term                  { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); }
           | vector_expr '+' vector_term      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str); }
           | vector_expr '-' vector_term      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str); }
           ;


vector_term: '[' scalars ']'                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "[" << $2.str << "]"); }
           | CENTROID '(' cell ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); }
           | CENTROID '(' face ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); }
           | NORMAL '(' face ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.normal(" << $3.str << ")"); }
           | '(' vector_expr ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); }              // produces 1 shift/reduce conflict
           | vector_term '*' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str); }             // produces 1 reduce/reduce conflict
           | scalar_factor '*' vector_term         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " * " << $3.str); }
           | vector_term '/' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " / " << $3.str); }
           | VARIABLE                              {
                                                      if(strcmp(getType($1.str), "vector") != 0)
                                                      {
                                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                      }
                                                      else
                                                      {
                                                          $$.str = strdup($1.str);
                                                      }
                                                   }

           | VARIABLE '(' values ')'               {
                                                      string st = functionToAnySingularType($1.str, "Vector", $3.str, "vector");
                                                      $$.str = strdup(st.c_str());
                                                   }
           ;


vectors: vector_term                      { $$.str = strdup($1.str); $$.array_size = 1; }
       | vectors ',' vector_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); $$.array_size = $1.array_size + 1; }
       ;


vector_exprs: vector_terms                       { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
            | '-' vector_terms                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "-" << $2.str); $$.grid_mapping = $2.grid_mapping; }            // produces 1 shift/reduce conflict
            | vector_exprs '+' vector_terms
                                                 {
                                                    if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                    {
                                                        LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " + " << $3.str);
                                                        $$.grid_mapping = $1.grid_mapping;
                                                    }
                                                 }

            | vector_exprs '-' vector_terms
                                                 {
                                                    if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                    {
                                                        LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str);
                                                        $$.grid_mapping = $1.grid_mapping;
                                                    }
                                                 }
            ;


vector_terms: '[' vectors ']'                        { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "[" << $2.str << "]"); $$.grid_mapping = $2.grid_mapping; }
            | CENTROID '(' cells ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
            | CENTROID '(' faces ')'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.centroid(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
            | NORMAL '(' faces ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.normal(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
            | '(' vector_exprs ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); $$.grid_mapping = $2.grid_mapping; }          // produces 1 shift/reduce conflict
            | vector_terms '*' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str); $$.grid_mapping = $1.grid_mapping; }         // produces 1 reduce/reduce conflict
            | scalar_factor '*' vector_terms         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str); $$.grid_mapping = $3.grid_mapping; }
            | vector_terms '/' scalar_factor         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " - " << $3.str); $$.grid_mapping = $1.grid_mapping; }
            | VARIABLE                               {
                                                        if(strcmp(getType($1.str), "vectors") != 0)
                                                        {
                                                            WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                        }
                                                        else
                                                        {
                                                            $$.str = strdup($1.str);
                                                            $$.grid_mapping = getSize1($1.str);
                                                        }
                                                     }

            | VARIABLE '(' values ')'                {
                                                          string st = functionToAnyCollectionType($1.str, "CollOfVectors", $3.str, "vectors");
                                                          if(st == "ok")
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                              $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup(st.c_str());
                                                          }
                                                     }
            ;


vertex: VARIABLE           {
                              if(strcmp(getType($1.str), "vertex") != 0)
                              {
                                  WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                              }
                              else
                              {
                                  $$.str = strdup($1.str);
                              }
                           }

      | VARIABLE '(' values ')'               {
                                                  string st = functionToAnySingularType($1.str, "Vertex", $3.str, "vertex");
                                                  $$.str = strdup(st.c_str());
                                              }
      ;


vertices: INTERIOR_VERTICES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorVertices()"); $$.grid_mapping = GRID_MAPPING_INTERIORVERTICES; }
        | BOUNDARY_VERTICES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryVertices()"); $$.grid_mapping = GRID_MAPPING_BOUNDARYVERTICES; }
        | ALL_VERTICES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allVertices()"); $$.grid_mapping = GRID_MAPPING_ALLVERTICES; }
        | VARIABLE                            {
                                                  if(strcmp(getType($1.str), "vertices") != 0)
                                                  {
                                                      WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                  }
                                                  else
                                                  {
                                                      $$.str = strdup($1.str);
                                                      $$.grid_mapping = getSize1($1.str);
                                                  }
                                              }

        | VARIABLE '(' values ')'                 {
                                                      string st = functionToAnyCollectionType($1.str, "CollOfVertices", $3.str, "vertices");
                                                      if(st == "ok")
                                                      {
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                          $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                      }
                                                      else
                                                      {
                                                          $$.str = strdup(st.c_str());
                                                      }
                                                  }
        ;


edge: VARIABLE             {
                              if(strcmp(getType($1.str), "edge") != 0)
                              {
                                  WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                              }
                              else
                              {
                                  $$.str = strdup($1.str);
                              }
                           }

    | VARIABLE '(' values ')'               {
                                                string st = functionToAnySingularType($1.str, "Edge", $3.str, "edge");
                                                $$.str = strdup(st.c_str());
                                            }
    ;


edges: INTERIOR_EDGES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorEdges()"); $$.grid_mapping = GRID_MAPPING_INTERIOREDGES; }
     | BOUNDARY_EDGES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryEdges()"); $$.grid_mapping = GRID_MAPPING_BOUNDARYEDGES; }
     | ALL_EDGES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allEdges()"); $$.grid_mapping = GRID_MAPPING_ALLEDGES; }
     | VARIABLE                         {
                                            if(strcmp(getType($1.str), "edges") != 0)
                                            {
                                                WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1.str);
                                                $$.grid_mapping = getSize1($1.str);
                                            }
                                        }

     | VARIABLE '(' values ')'                  {
                                                    string st = functionToAnyCollectionType($1.str, "CollOfEdges", $3.str, "edges");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                        $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
     ;


face: VARIABLE                    {
                                      if(strcmp(getType($1.str), "face") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1.str);
                                      }
                                  }

    | VARIABLE '(' values ')'               {
                                                string st = functionToAnySingularType($1.str, "Face", $3.str, "face");
                                                $$.str = strdup(st.c_str());
                                            }
    ;


faces: INTERIOR_FACES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorFaces()"); $$.grid_mapping = GRID_MAPPING_INTERIORFACES; }
     | BOUNDARY_FACES '(' GRID ')'      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryFaces()"); $$.grid_mapping = GRID_MAPPING_BOUNDARYFACES; }
     | ALL_FACES '(' GRID ')'           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allFaces()"); $$.grid_mapping = GRID_MAPPING_ALLFACES; }
     | VARIABLE                         {
                                            if(strcmp(getType($1.str), "faces") != 0)
                                            {
                                                WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                            }
                                            else
                                            {
                                                $$.str = strdup($1.str);
                                                $$.grid_mapping = getSize1($1.str);
                                            }
                                        }

     | VARIABLE '(' values ')'                   {
                                                      string st = functionToAnyCollectionType($1.str, "CollOfFaces", $3.str, "faces");
                                                      if(st == "ok")
                                                      {
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                          $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                      }
                                                      else
                                                      {
                                                          $$.str = strdup(st.c_str());
                                                      }
                                                 }
     ;


cell: FIRST_CELL '(' face ')'     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.firstCell(" << $3.str << ")"); }
    | SECOND_CELL '(' face ')'    { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.secondCell(" << $3.str << ")"); }
    | VARIABLE                    {
                                      if(strcmp(getType($1.str), "cell") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1.str);
                                      }
                                  }

    | VARIABLE '(' values ')'                 {
                                                  string st = functionToAnySingularType($1.str, "Cell", $3.str, "cell");
                                                  $$.str = strdup(st.c_str());
                                              }
    ;


cells: INTERIOR_CELLS '(' GRID ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.interiorCells()"); $$.grid_mapping = GRID_MAPPING_INTERIORCELLS; }
     | BOUNDARY_CELLS '(' GRID ')'          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.boundaryCells()"); $$.grid_mapping = GRID_MAPPING_BOUNDARYCELLS;}
     | ALL_CELLS '(' GRID ')'               { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.allCells()"); $$.grid_mapping = GRID_MAPPING_ALLCELLS;}
     | FIRST_CELL '(' faces ')'             { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.firstCell(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
     | SECOND_CELL '(' faces ')'            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.secondCell(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
     | VARIABLE                             {
                                                if(strcmp(getType($1.str), "cells") != 0)
                                                {
                                                    WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                }
                                                else
                                                {
                                                    $$.str = strdup($1.str);
                                                    $$.grid_mapping = getSize1($1.str);
                                                }
                                            }

     | VARIABLE '(' values ')'                  {
                                                    string st = functionToAnyCollectionType($1.str, "CollOfCells", $3.str, "cells");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                        $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
     ;


adb: GRADIENT '(' adb ')'         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); }
   | DIVERGENCE '(' adb ')'       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); }
   | VARIABLE                     {
                                      if(strcmp(getType($1.str), "scalarAD") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1.str);
                                      }
                                  }

   | VARIABLE '(' values ')'                {
                                                string st = functionToAnySingularType($1.str, "ScalarAD", $3.str, "scalarAD");
                                                $$.str = strdup(st.c_str());
                                            }
   ;


adbs: GRADIENT '(' adbs ')'       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.gradient(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping; }
    | DIVERGENCE '(' adbs ')'     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "er.divergence(" << $3.str << ")"); $$.grid_mapping = $3.grid_mapping;}
    | VARIABLE                    {
                                      if(strcmp(getType($1.str), "scalarsAD") != 0)
                                      {
                                          WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                      }
                                      else
                                      {
                                          $$.str = strdup($1.str);
                                          $$.grid_mapping = getSize1($1.str);
                                      }
                                  }

    | VARIABLE '(' values ')'                   {
                                                    string st = functionToAnyCollectionType($1.str, "CollOfScalarsAD", $3.str, "scalarsAD");
                                                    if(st == "ok")
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                        $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                    }
                                                    else
                                                    {
                                                        $$.str = strdup(st.c_str());
                                                    }
                                                }
    ;


boolean_expr: boolean_term                           { $$.str = strdup($1.str); }
            | NOT boolean_term                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "!" << $2.str); }
            | boolean_expr AND boolean_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " && " << $3.str); }
            | boolean_expr OR boolean_term           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " || " << $3.str); }
            | boolean_expr XOR boolean_term          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(!" << $1.str << " && " << $3.str << ") || (!" << $3.str << " && " << $1.str << ")"); }
            ;



boolean_term: TRUE                                   { $$.str = strdup("true"); }
            | FALSE                                  { $$.str = strdup("false"); }
            | scalar_expr '>' scalar_expr            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " > " << $3.str); }
            | scalar_expr '<' scalar_expr            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " < " << $3.str); }
            | scalar_expr LESSEQ scalar_expr         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " <= " << $3.str); }
            | scalar_expr GREATEREQ scalar_expr      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " >= " << $3.str); }
            | scalar_expr EQ scalar_expr             { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " == " << $3.str); }
            | scalar_expr NOTEQ scalar_expr          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " != " << $3.str); }
            | '(' boolean_expr ')'                   { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); }
            | VARIABLE                               {
                                                        if(strcmp(getType($1.str), "bool") != 0)
                                                        {
                                                            WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                        }
                                                        else
                                                        {
                                                            $$.str = strdup($1.str);
                                                        }
                                                    }

            | VARIABLE '(' values ')'               {
                                                        string st = functionToAnySingularType($1.str, "bool", $3.str, "bool");
                                                        $$.str = strdup(st.c_str());
                                                    }
            ;


boolean_exprs: boolean_terms                           { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
             | NOT boolean_terms                       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "!" << $2.str); $$.grid_mapping = $2.grid_mapping;}
             | boolean_exprs AND boolean_terms
                                                       {
                                                          if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " && " << $3.str);
                                                              $$.grid_mapping = $1.grid_mapping;
                                                          }
                                                       }

             | boolean_exprs OR boolean_terms
                                                       {
                                                          if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " || " << $3.str);
                                                              $$.grid_mapping = $1.grid_mapping;
                                                          }
                                                       }

             | boolean_exprs XOR boolean_terms
                                                       {
                                                          if($1.grid_mapping != $3.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(!" << $1.str << " && " << $3.str << ") || (!" << $3.str << " && " << $1.str << ")");
                                                              $$.grid_mapping = $1.grid_mapping;
                                                          }
                                                       }
             ;



boolean_terms: '(' scalars ')' '>' '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") > (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }

             | '(' scalars ')' '<' '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") < (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }

             | '(' scalars ')' LESSEQ '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") <= (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }

             | '(' scalars ')' GREATEREQ '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") >= (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }

             | '(' scalars ')' EQ '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") == (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }

             | '(' scalars ')' NOTEQ '(' scalars ')'
                                                      {
                                                          if($2.grid_mapping != $6.grid_mapping)    // check that the lengths of the 2 terms are equal
                                                          {
                                                              LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$.str);
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ") != (" << $6.str << ")");
                                                              $$.grid_mapping = $2.grid_mapping;
                                                          }
                                                      }
             | '(' boolean_exprs ')'                  { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "(" << $2.str << ")"); $$.grid_mapping = $2.grid_mapping;}
             | VARIABLE                               {
                                                          if(strcmp(getType($1.str), "bools") != 0)
                                                          {
                                                              WRONG_TYPE_ERROR_TO_CHAR_ARRAY($$.str, $1.str);
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup($1.str);
                                                              $$.grid_mapping = getSize1($1.str);
                                                          }
                                                      }

             | VARIABLE '(' values ')'                {
                                                          string st = functionToAnyCollectionType($1.str, "CollOfBools", $3.str, "bools");
                                                          if(st == "ok")
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << "(" << $3.str << ")");
                                                              $$.grid_mapping = fun[getIndex2($1.str)].grid_mapping;
                                                          }
                                                          else
                                                          {
                                                              $$.str = strdup(st.c_str());
                                                          }
                                                      }
             ;



plural: scalar_exprs            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | vector_exprs            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | vertices                { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | edges                   { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | faces                   { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | cells                   { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | adbs                    { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      | boolean_exprs           { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; }
      ;


header: VARIABLE HEADER_DECL SCALAR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Scalar " << $1.str); }
      | VARIABLE HEADER_DECL VECTOR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Vector " << $1.str); }
      | VARIABLE HEADER_DECL VERTEX                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Vertex " << $1.str); }
      | VARIABLE HEADER_DECL EDGE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Edge " << $1.str); }
      | VARIABLE HEADER_DECL FACE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Face " << $1.str); }
      | VARIABLE HEADER_DECL CELL                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "Cell " << $1.str); }
      | VARIABLE HEADER_DECL ADB                             { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "ScalarAD " << $1.str); }
      | VARIABLE HEADER_DECL BOOLEAN                         { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "bool " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF SCALAR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfScalars " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF VECTOR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfVectors " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF VERTEX            { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfVertices " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF EDGE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfEdges " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF FACE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfFaces " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF CELL              { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfCells " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF ADB               { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfScalarsAD " << $1.str); }
      | VARIABLE HEADER_DECL COLLECTION OF BOOLEAN           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "CollOfBools " << $1.str); }
      ;


parameter_list: header                         { $$.str = strdup($1.str); }
              | parameter_list ',' header      { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); }
              ;


commands: command1                              { $$.str = strdup($1.str); }
        | commands end_lines command1           { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << $2.str << $3.str); }
        |                                       { $$.str = strdup(""); }     // a function can have only the return instruction
        ;


type: SCALAR                                { $$.str = strdup("Scalar"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | VECTOR                                { $$.str = strdup("Vector"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | VERTEX                                { $$.str = strdup("Vertex"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | EDGE                                  { $$.str = strdup("Edge"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | FACE                                  { $$.str = strdup("Face"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | CELL                                  { $$.str = strdup("Cell"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | ADB                                   { $$.str = strdup("ScalarAD"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | BOOLEAN                               { $$.str = strdup("bool"); $$.grid_mapping = GRID_MAPPING_ENTITY; }
    | COLLECTION OF SCALAR                  { $$.str = strdup("CollOfScalars"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF VECTOR                  { $$.str = strdup("CollOfVectors"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF VERTEX                  { $$.str = strdup("CollOfVertices"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF EDGE                    { $$.str = strdup("CollOfEdges"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF FACE                    { $$.str = strdup("CollOfFaces"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF CELL                    { $$.str = strdup("CollOfCells"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF ADB                     { $$.str = strdup("CollOfScalarsAD"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF BOOLEAN                 { $$.str = strdup("CollOfBools"); $$.grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF SCALAR ON plural        { $$.str = strdup("CollOfScalars"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF VECTOR ON plural        { $$.str = strdup("CollOfVectors"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF VERTEX ON plural        { $$.str = strdup("CollOfVertices"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF EDGE ON plural          { $$.str = strdup("CollOfEdges"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF FACE ON plural          { $$.str = strdup("CollOfFaces"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF CELL ON plural          { $$.str = strdup("CollOfCells"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF ADB ON plural           { $$.str = strdup("CollOfScalarsAD"); $$.grid_mapping = $5.grid_mapping; }
    | COLLECTION OF BOOLEAN ON plural       { $$.str = strdup("CollOfBools"); $$.grid_mapping = $5.grid_mapping; }
    ;


//////////////////////////////////////////////////////////////////////// these support input parameters as expressions with or without ON (option 1)
/*
value: scalar            {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | vector            {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | vertex            {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | edge              {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | face              {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | cell              {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | adb               {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | boolean           {$$.str = strdup($1.str); $$.grid_mapping = 1;}
     | scalar_exprs      {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | vector_exprs      {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | vertices          {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | edges             {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | faces             {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | cells             {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | adbs              {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     | booleans          {$$.str = strdup($1.grid_mapping); $$.grid_mapping = $1.grid_mapping;}
     ;


values: value                   {$$.str = strdup($1.str); itoa($$.grid_mappings, $1.grid_mappings, 100);}
      | values ',' value        {
                                  char *str = append5($1.str,',',$3.str);
                                  $$.str = strdup(str);
                                  free(str);
                                  char *temp = (char *)malloc(1000 * sizeof(char));
                                  itoa(temp, $3.grid_mapping, 100);
                                  char *str2 = append5($1.grid_mappings,',',temp);
                                  $$.grid_mappings = strdup(str2);
                                  free(str2);
                                }
      ;
*/


//////////////////////////////////////////////////////////////////////// these support input parameters as expressions without ON (option 2)
/*
value: scalar_expr       {$$.str = strdup($1.str);}
     | vector_expr       {$$.str = strdup($1.str);}
     | vertex            {$$.str = strdup($1.str);}
     | edge              {$$.str = strdup($1.str);}
     | face              {$$.str = strdup($1.str);}
     | cell              {$$.str = strdup($1.str);}
     | adb               {$$.str = strdup($1.str);}
     | boolean_expr      {$$.str = strdup($1.str);}
     | scalar_exprs      {$$.str = strdup($1.str);}
     | vector_exprs      {$$.str = strdup($1.str);}
     | vertices          {$$.str = strdup($1.str);}
     | edges             {$$.str = strdup($1.str);}
     | faces             {$$.str = strdup($1.str);}
     | cells             {$$.str = strdup($1.str);}
     | adbs              {$$.str = strdup($1.str);}
     | boolean_exprs     {$$.str = strdup($1.str);}
     ;


// we need 'values' to be a structure with 2 strings: one which will store the exact output which should be displayed, and another which should store all the terms separated by an unique character ('@')
values: value                   {$$.cCode = strdup($1.str); $$.sepCode = strdup($1.str);}
      | values ',' value        {char *str = append5($1.str.cCode,',',$3.str); $$.cCode = strdup(str); free(str); $$.sepCode = append5($1.str.sepCode, '@', $3.str);}
      ;
*/


//////////////////////////////////////////////////////////////////////// this supports input parameters as variables
values: VARIABLE                { $$.str = strdup($1.str); }
      | values ',' VARIABLE     { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << ", " << $3.str); }
      ;


end_lines: '\n'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "\n"); currentLineNumber++; }
         | '\n' end_lines       { STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "\n" << $2.str); currentLineNumber++; }
         |                      { $$.str = strdup(""); }
         ;


return_instr: RETURN boolean_expr '?' VARIABLE ':' VARIABLE
                  {
                    if(check5($4.str) == false || check5($6.str) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.grid_mapping = GRID_MAPPING_INVALID;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "return " << $2.str << " ? " << $4.str << " : " << $6.str);
                        $$.grid_mapping = getSize3($4.str);
                    }
                  }

            | RETURN VARIABLE
                  {
                    if(check5($2.str) == false)
                    {
                        $$.str = strdup("Invalid");
                        $$.grid_mapping = GRID_MAPPING_INVALID;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "return " << $2.str << ";");
                        $$.grid_mapping = getSize3($2.str);
                    }
                  }
            ;


function_start: VARIABLE '=' end_lines '{'
                                            {
                                              insideFunction = true;
                                              currentFunctionIndex = getIndex2($1.str);
                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << " = " << $3.str << "{");
                                            }


// these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: VARIABLE ':' FUNCTION '(' parameter_list ')' RET type
                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), $1.str) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "error at line " << currentLineNumber << ": The function '" << $1.str << "' is redeclared");
                                                }
                                                else
                                                {
                                                        fun[funNo++].name = strdup($1.str);
                                                        fun[funNo-1].returnType = CPPToEquelle1($8.str);
                                                        fun[funNo-1].grid_mapping = $8.grid_mapping;
                                                        fun[funNo-1].noLocalVariables = 0;
                                                        fun[funNo-1].noParam = 0;

                                                        char *cs1 = strdup($5.str);    // we need to make a copy, because the strtok function modifies the given string
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
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].grid_mapping = CPPToEquelle2(copy);  // the string we have as a parameter list is already transformed in C++, but we need the types' lengths
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].assigned = true;
                                                          fun[funNo-1].signature = strdup($5.str);

                                                          pch = strtok(NULL, ",");
                                                        }

                                                        fun[funNo-1].assigned = false;
                                                        // STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $8.str << " " << $1.str << "(" << $5.str << ")" << ";");
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "");
                                                }
                                            }
                    ;


function_assignment: function_start end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional

                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), extract($1.str)) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                      if(fun[i].assigned == true)
                                                      {
                                                          stringstream ss;
                                                          ss << "error at line " << currentLineNumber << ": The function '" << fun[i].name << "' is reassigned";
                                                          $$.str = strdup(ss.str().c_str());
                                                      }
                                                      else
                                                      {
                                                          if($5.grid_mapping != GRID_MAPPING_INVALID)
                                                          {
                                                              // STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, $1.str << $2.str << $3.str << $4.str << $5.str << $6.str << "}");
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "auto " << fun[i].name << "[&](" << fun[i].signature << ") -> " << EquelleToCPP(fun[i].returnType) << " {\n" << $2.str << $3.str << $4.str << $5.str << $6.str << "}");
                                                              if(fun[i].grid_mapping == GRID_MAPPING_ANY && $5.grid_mapping != GRID_MAPPING_ANY)
                                                                  fun[i].grid_mapping = $5.grid_mapping;
                                                              else
                                                                  if(fun[i].grid_mapping != GRID_MAPPING_ANY && $5.grid_mapping == GRID_MAPPING_ANY)
                                                                      {;}   // do nothing (the function must keep its return size from the definition)
                                                                  else
                                                                      {;}   // if both are ANY, the function's return type is already correct; if none are ANY, then they should already be equal, otherwise the instruction flow wouldn't enter on this branch
                                                              fun[i].assigned = true;
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "error at line " << currentLineNumber << ": At least one of the return variables does not exist within the function or the return type of the function '" << fun[i].name << "' from its assignment differs than the length of the return type from the function's definition");
                                                          }

                                                      }
                                                else
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$.str, "error at line " << currentLineNumber << ": The function '" << extract($1.str) <<"' must be declared before being assigned");
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






output: OUTPUT '(' VARIABLE ')'       { string out = output_function($3.str); $$.str = strdup(out.c_str()); }








singular_declaration: VARIABLE ':' SCALAR               { string out = singular_declaration_function($1.str, "scalar"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' VECTOR               { string out = singular_declaration_function($1.str, "vector"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' VERTEX               { string out = singular_declaration_function($1.str, "vertex"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' EDGE                 { string out = singular_declaration_function($1.str, "edge"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' FACE                 { string out = singular_declaration_function($1.str, "face"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' CELL                 { string out = singular_declaration_function($1.str, "cell"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' ADB                  { string out = singular_declaration_function($1.str, "scalarAD"); $$.str = strdup(out.c_str()); }
                    | VARIABLE ':' BOOLEAN              { string out = singular_declaration_function($1.str, "bool"); $$.str = strdup(out.c_str()); }
                    ;


plural_declaration: VARIABLE ':' COLLECTION OF SCALAR       { string out = plural_declaration_function($1.str, "scalars"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF VECTOR       { string out = plural_declaration_function($1.str, "vectors"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF VERTEX       { string out = plural_declaration_function($1.str, "vertices"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF EDGE         { string out = plural_declaration_function($1.str, "edges"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF FACE         { string out = plural_declaration_function($1.str, "faces"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF CELL         { string out = plural_declaration_function($1.str, "cells"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF ADB          { string out = plural_declaration_function($1.str, "scalarsAD"); $$.str = strdup(out.c_str()); }
                  | VARIABLE ':' COLLECTION OF BOOLEAN      { string out = plural_declaration_function($1.str, "bools"); $$.str = strdup(out.c_str()); }
                  ;


extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "scalars", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF VECTOR ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "vectors", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF VERTEX ON plural      { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "vertices", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF EDGE ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "edges", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF FACE ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "faces", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF CELL ON plural        { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "cells", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF ADB ON plural         { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "scalarsAD", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural     { char *st = strdup($7.str); string out = extended_plural_declaration_function($1.str, "bools", st, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                           ;


declaration: singular_declaration           { char* out = strdup($1.str); $$.str = out; }
           | plural_declaration             { char* out = strdup($1.str); $$.str = out; }
           | extended_plural_declaration    { char* out = strdup($1.str); $$.str = out; }
           ;


expression: scalar_expr       { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_SCALAR; $$.collection = false; }
          | vector_expr       { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_VECTOR; $$.collection = false; }
          | vertex            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_VERTEX; $$.collection = false; }
          | edge              { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_EDGE; $$.collection = false; }
          | face              { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_FACE; $$.collection = false; }
          | cell              { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_CELL; $$.collection = false; }
          | boolean_expr      { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_BOOLEAN; $$.collection = false; }
          ;


expressions: scalar_exprs     { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_SCALAR; $$.collection = true; }
           | vector_exprs     { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_VECTOR; $$.collection = true; }
           | vertices         { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_VERTEX; $$.collection = true; }
           | edges            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_EDGE; $$.collection = true; }
           | faces            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_FACE; $$.collection = true; }
           | cells            { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_CELL; $$.collection = true; }
           | boolean_exprs    { $$.str = strdup($1.str); $$.grid_mapping = $1.grid_mapping; $$.array_size = $1.array_size; $$.type = TYPE_BOOLEAN; $$.collection = true; }
           ;


singular_assignment: VARIABLE '=' USS                      { string out = USS_assignment_function($1.str); $$.str = strdup(out.c_str()); }
                   | VARIABLE '=' USSWD '(' number ')'     { string out = USSWD_assignment_function($1.str, $5.str); $$.str = strdup(out.c_str()); }
                   | VARIABLE '=' expression               { string str2 = getVariableTypeString1($3.type, false); string str1 = getVariableTypeString2($3.type, false); string out = singular_assignment_function($1.str, strdup(str1.c_str()), $3.str, strdup(str2.c_str())); $$.str = strdup(out.c_str()); }
                   ;


plural_assignment: VARIABLE '=' USCOS '(' plural ')'      { string out = USCOS_assignment_function($1.str, $5.str, $5.grid_mapping); $$.str = strdup(out.c_str()); }
                 | VARIABLE '=' expressions               { string str2 = getVariableTypeString1($3.type, true); string str1 = getVariableTypeString2($3.type, true); string out = plural_assignment_function($1.str, strdup(str1.c_str()), $3.str, strdup(str2.c_str()), $3.grid_mapping); $$.str = strdup(out.c_str()); }
                 ;


//if the variable hasn't been declared before, it is an assignment with deduced declaration (type)

assignment: singular_assignment     { char* out = strdup($1.str); $$.str = out; }
          | plural_assignment       { char* out = strdup($1.str); $$.str = out; }
          ;




singular_declaration_with_assignment: VARIABLE ':' SCALAR '=' scalar_expr          { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "scalar", st, "Scalar"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' VECTOR '=' vector_expr          { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "vector", st, "Vector"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' VERTEX '=' vertex               { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "vertex", st, "Vertex"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' EDGE '=' edge                   { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "edge", st, "Edge"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' FACE '=' face                   { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "face", st, "Face"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' CELL '=' cell                   { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "cell", st, "Cell"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' ADB '=' adb                     { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "scalarAD", st, "ScalarAD"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' BOOLEAN '=' boolean_expr        { char *st = strdup($5.str); string out = singular_declaration_with_assignment_function($1.str, "bool", st, "bool"); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' SCALAR '=' USS                  { string out = USS_declaration_with_assignment_function($1.str); $$.str = strdup(out.c_str()); }
                                    | VARIABLE ':' SCALAR '=' USSWD '(' number ')' { string out = USSWD_declaration_with_assignment_function($1.str, $7.str); $$.str = strdup(out.c_str()); }
                                    ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' scalar_exprs          { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "scalars", st, "CollOfScalars", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF VECTOR '=' vector_exprs          { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "vectors", st, "CollOfVectors", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF VERTEX '=' vertices              { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "vertices", st, "CollOfVertices", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF EDGE '=' edges                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "edges", st, "CollOfEdges", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF FACE '=' faces                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "faces", st, "CollOfFaces", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF CELL '=' cells                   { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "cells", st, "CollOfCells", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF ADB '=' adbs                     { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "scalarsAD", st, "CollOfScalarsAD", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF BOOLEAN '=' boolean_exprs        { char *st = strdup($7.str); string out = plural_declaration_with_assignment_function($1.str, "bools", st, "CollOfBools", $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                  | VARIABLE ':' COLLECTION OF SCALAR '=' USCOS '(' plural ')'  { string out = USCOS_declaration_with_assignment_function($1.str, $9.str, $9.grid_mapping); $$.str = strdup(out.c_str()); }
                                  ;


extended_plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR ON plural '=' scalar_exprs          { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "scalars", st1, "CollOfScalars", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF VECTOR ON plural '=' vector_exprs          { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "vectors", st1, "CollOfVectors", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF VERTEX ON plural '=' vertices              { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "vertices", st1, "CollOfVertices", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF EDGE ON plural '=' edges                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "edges", st1, "CollOfEdges", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF FACE ON plural '=' faces                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "faces", st1, "CollOfFaces", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF CELL ON plural '=' cells                   { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "cells", st1, "CollOfCells", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF ADB ON plural '=' adbs                     { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "scalarsAD", st1, "CollOfScalarsAD", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF BOOLEAN ON plural '=' boolean_exprs        { char *st1 = strdup($9.str); char *st2 = strdup($7.str); string out = extended_plural_declaration_with_assignment_function($1.str, "bools", st1, "CollOfBools", st2, $9.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           | VARIABLE ':' COLLECTION OF SCALAR ON plural '=' USCOS '(' plural ')'  { string out = USCOS_extended_declaration_with_assignment_function($1.str, $11.str, $7.str, $11.grid_mapping, $7.grid_mapping); $$.str = strdup(out.c_str()); }
                                           ;



 declaration_with_assignment: singular_declaration_with_assignment          { char* out = strdup($1.str); $$.str = out; }
                            | plural_declaration_with_assignment            { char* out = strdup($1.str); $$.str = out; }
                            | extended_plural_declaration_with_assignment   { char* out = strdup($1.str); $$.str = out; }
                            ;




// instructions which can be used in the program and in a function's body
command: declaration                    { char* out = strdup($1.str); $$.str = out; }
       | assignment                     { char* out = strdup($1.str); $$.str = out; }
       | declaration_with_assignment    { char* out = strdup($1.str); $$.str = out; }
       ;


command1: command                       { char* out = strdup($1.str); $$.str = out; }
        | command COMMENT               { string st1 = $1.str; string st2 = $2.str; stringstream ss; ss << st1 << " // " << st2.substr(1, st2.size() - 1); $$.str = strdup(ss.str().c_str()); }
        | COMMENT                       { string st1 = $1.str; stringstream ss; ss << "// " << st1.substr(1, st1.size() - 1); $$.str = strdup(ss.str().c_str()); }
        ;


// instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                    { stringstream ss; ss << $1.str; $$.str = strdup(ss.str().c_str()); }
        | function_declaration                       { stringstream ss; ss << $1.str; $$.str = strdup(ss.str().c_str()); }
        | function_assignment                        { stringstream ss; ss << $1.str; $$.str = strdup(ss.str().c_str()); }
        | output                                     { stringstream ss; ss << $1.str; $$.str = strdup(ss.str().c_str()); }
    //  | function_declaration_with_assignment       { stringstream ss; ss << $1.str; $$.str = strdup(ss.str().c_str()); }
        ;


pr: pr command2 '\n'                  {
                                        string out = $2.str;
                                        cout << out << endl;
                                        currentLineNumber++;
                                      }
  | pr command2 COMMENT '\n'          {
                                        string out1 = $2.str;
                                        string out2 = $3.str;
                                        cout << out1 << " // " << out2.substr(1, out2.size() - 1) << endl;   //+1 to skip comment sign (#)
                                        currentLineNumber++;
                                      }
  | pr COMMENT '\n'                   {
                                        string out = $2.str;
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
    if(strcmp(fun[currentFunctionIndex].headerVariables[i].type.c_str(), fun[currentFunctionIndex].returnType.c_str()) != 0 || (fun[currentFunctionIndex].headerVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY && fun[currentFunctionIndex].headerVariables[i].grid_mapping != GRID_MAPPING_ANY))
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
    if(strcmp(fun[currentFunctionIndex].localVariables[i].type.c_str(), fun[currentFunctionIndex].returnType.c_str()) != 0 || (fun[currentFunctionIndex].localVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY && fun[currentFunctionIndex].localVariables[i].grid_mapping != GRID_MAPPING_ANY))
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
GridMapping getSize1(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(s1, var[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return var[i].grid_mapping;
      }
  }
  HEAP_CHECK();
  return GRID_MAPPING_INVALID;
}


// function which returns the return size of a function, based on its name
GridMapping getSize2(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < funNo; i++)
  {
      if(strcmp(s1, fun[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[i].grid_mapping;
      }
  }
  HEAP_CHECK();
  return GRID_MAPPING_INVALID;
}


// function which returns the size of a header/local variable inside the current function, based on its name
GridMapping getSize3(char* s1)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[currentFunctionIndex].headerVariables[i].grid_mapping;
      }
  }
  for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
  {
      if(strcmp(s1, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return fun[currentFunctionIndex].localVariables[i].grid_mapping;
      }
  }
  HEAP_CHECK();
  return GRID_MAPPING_INVALID;
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
GridMapping CPPToEquelle2(char* st)
{
    if(strcmp(st, "Scalar") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Vector") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Vertex") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Edge") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Face") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Cell") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "ScalarAD") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "bool") == 0) {
      return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "CollOfScalars") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfVectors") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfVertices") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfEdges") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfCells") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfScalarsAD") == 0) {
      return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfBools") == 0) {
      return GRID_MAPPING_ANY;
    }

    return GRID_MAPPING_INVALID;
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
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
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
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ANY;
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
              var[varNo-1].grid_mapping = GRID_MAPPING_ANY;
              var[varNo-1].assigned = false;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string extended_plural_declaration_function(char* st1, char* st2, char* st3, GridMapping d1)
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
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = d1;
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
                          var[varNo-1].grid_mapping = d1;
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
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
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
                                    var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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


string plural_assignment_function(char* st1, char* st2, char* st3, char* st4, GridMapping d1)
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
                                                    if(getSize3(st1) == GRID_MAPPING_ANY)
                                                    {
                                                        fun[currentFunctionIndex].localVariables[i].grid_mapping = d1;
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
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = d1;
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
                                                if(getSize1(st1) == GRID_MAPPING_ANY)
                                                {
                                                    var[i].grid_mapping = d1;
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
                                    var[varNo-1].grid_mapping = d1;
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
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
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
                                    var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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


string plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, GridMapping d1)
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
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = d1;
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
                                    var[varNo-1].grid_mapping = d1;
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


string extended_plural_declaration_with_assignment_function(char* st1, char* st2, char* st3, char* st4, char* st5, GridMapping d1, GridMapping d2)
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
                                                        fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = d1;
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
                                                  var[varNo-1].grid_mapping = d1;
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
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
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
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_assignment_function(char* st1, char* st2, GridMapping d1)
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
                                                if(getSize1(st1) == GRID_MAPPING_ANY)
                                                {
                                                    var[i].grid_mapping = d1;
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
                                    var[varNo-1].grid_mapping = d1;
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


string USCOS_declaration_with_assignment_function(char* st1, char* st2, GridMapping d1)
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
                                    var[varNo-1].grid_mapping = d1;
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


string USCOS_extended_declaration_with_assignment_function(char* st1, char* st2, char* st3, GridMapping d1, GridMapping d2)
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
                                                  var[varNo-1].grid_mapping = d1;
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


string getVariableTypeString1(VariableType v, bool collection)
{
	std::stringstream ss;

	switch(v) {
		case TYPE_SCALAR:
			ss << ((collection) ? "CollOfScalars" : "Scalar");
			break;
		case TYPE_VECTOR:
			ss << ((collection) ? "CollOfVectors" : "Vector");
			break;
		case TYPE_VERTEX:
			ss << ((collection) ? "CollOfVertices" : "Vertex");
			break;
		case TYPE_EDGE:
			ss << ((collection) ? "CollOfEdges" : "Edge");
			break;
		case TYPE_FACE:
			ss << ((collection) ? "CollOfFaces" : "Face");
			break;
		case TYPE_CELL:
			ss << ((collection) ? "CollOfCells" : "Cell");
			break;
		case TYPE_BOOLEAN:
			ss << ((collection) ? "CollOfBools" : "bool");
			break;
		case TYPE_INVALID:
			ss << ((collection) ? "CollOfInvalidTypes" : "InvalidType");
			break;
		default:
			ss << ((collection) ? "CollOfUnknownTypes" : "UnknownType");
	}

	return ss.str();
}


string getVariableTypeString2(VariableType v, bool collection)
{
  std::stringstream ss;

  switch(v) {
    case TYPE_SCALAR:
      ss << ((collection) ? "scalars" : "scalar");
      break;
    case TYPE_VECTOR:
      ss << ((collection) ? "vectors" : "vector");
      break;
    case TYPE_VERTEX:
      ss << ((collection) ? "vertices" : "vertex");
      break;
    case TYPE_EDGE:
      ss << ((collection) ? "edges" : "edge");
      break;
    case TYPE_FACE:
      ss << ((collection) ? "faces" : "face");
      break;
    case TYPE_CELL:
      ss << ((collection) ? "cells" : "cell");
      break;
    case TYPE_BOOLEAN:
      ss << ((collection) ? "bools" : "bool");
      break;
    case TYPE_INVALID:
      ss << ((collection) ? "CollOfInvalidTypes" : "InvalidType");
      break;
    default:
      ss << ((collection) ? "CollOfUnknownTypes" : "UnknownType");
  }

  return ss.str();
}
