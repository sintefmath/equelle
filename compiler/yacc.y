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

	#define STREAM_TO_DOLLARS_CHAR_ARRAY(dd, streamcontent)                 do { stringstream ss; ss << streamcontent; dd = ss.str(); } while (false)
	#define LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY(dd)           do { stringstream ss; ss << "length_mismatch_error"; dd = ss.str(); } while (false)
	// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
	#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)                          do { stringstream ss; ss << "wrong_type_error  " << d1; dd = ss.str(); }  while (false)
	// we print the name of the variable too in order to prioritize the error checking of variable name included in its own definition over the "wrong type variable" error
	#define WRONG_TYPE_ERROR_TO_CHAR_ARRAY(dd, d1)            do { stringstream ss; ss << "wrong_type_error  " << d1; dd = ss.str(); }  while (false)

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
	  string name;           // must begin with a small letter
	  VariableType type;           // can be: scalar, vector, vertex, scalars etc.
	  GridMapping grid_mapping; // if the type is a singular type, then the length is 1; otherwise it can be any other number >= 1
	  bool assigned;         // we want to know if a variable has been assigned, in order to prevent errors (example: operations with unassigned variables)
	};


	// global structure and counter for storing the names of the functions
	struct FunctionStructureForCPP
	{
	  string name;                                      // g1
	  VariableType type;                                // TYPE_SCALAR
	  GridMapping grid_mapping;                         // GRID_MAPPING_ALLCELLS
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
	void yyerror(const const char* s);
	int yylex(void);
	bool find1(const char* s1, const char* s2);
	char* find2(const char* s1);
	char* find3(const char* s1);
	int find4(const char* s1);
	char* find5(const char* s1);
	char* find6(const char* s1);
	bool check1(const char* s1);
	bool check2(const char* s1);
	bool check3(const char* s1);
	bool check4(const char* s1);
	bool check5(const char* s1);
	bool check6(const char* s1);
	bool check7(const char* s1);
	bool check8(const char* s1, const char* s2);
	string check9(const char* s1);
	VariableType getType(const char* variable_name);
	int getIndex1(const char* s1);
	int getIndex2(const char* s1);
	GridMapping getSize1(const char* s1);
	GridMapping getSize2(const char* s1);
	GridMapping getSize3(const char* s1);
	int getSize4(const char* s1);
	char* extract(const char* s1);
	VariableType getVariableType(const char* st);
	GridMapping getGridMapping(const char* st);
	string errorTypeToErrorMessage(string errorType);
	string functionToAnySingularType(const char *st1, const char *st2, const char *st3, const string &st4);
	string functionToAnyCollectionType(const char *st1, const char *st2, const char *st3, const string &st4);
	string declaration_function(const char* variable_name, EntityType entity, bool collection);
	string extended_plural_declaration_function(const char* variable_name, EntityType entity, const char* ON_expression, GridMapping ON_expression_grid_mapping);
	string singular_assignment_function(const char* variable_name, const info* right_hand_side);
	string plural_assignment_function(const char* variable_name, const info* right_hand_side);
	string declaration_with_assignment_function(const char* variable_name, const info* right_hand_side);
	string extended_plural_declaration_with_assignment_function(const char* variable_name, const info* rhs, const GridMapping& lhs);
	string USS_assignment_function(const char* st1);
	string USS_declaration_with_assignment_function(const char* st1);
	string USSWD_assignment_function(const char* st1, const char* st2);
	string USSWD_declaration_with_assignment_function(const char* st1, const char* st2);
	string USCOS_assignment_function(const char* st1, const char* st2, GridMapping d1);
	string USCOS_declaration_with_assignment_function(const char* st1, const char* st2, GridMapping d1);
	string USCOS_extended_declaration_with_assignment_function(const char* st1, const char* st2, const char* st3, GridMapping d1, GridMapping d2);
	string output_function(std::string& st1);
	string getStringFromVariableType(VariableType variable);
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


bool operator==(const VariableType& a, const VariableType& b) {
	if (a.entity_type == b.entity_type && a.collection == b.collection) {
		return true;
	}
	return false;
}

bool operator!=(const VariableType& a, const VariableType& b) {
	return !(a == b);
}

} //Code


%type<inf> floating_point
%type<inf> number
%type<inf> scalars
%type<inf> vectors
%type<inf> expression
%type<inf> INTEGER
%type<inf> VARIABLE
%type<inf> COMMENT
/*
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
%type<inf> command1
*/
%type<inf> command
%type<inf> command2
%type<inf> singular_declaration
%type<inf> plural_declaration
%type<inf> extended_plural_declaration
%type<inf> singular_declaration_with_assignment
%type<inf> plural_declaration_with_assignment
%type<inf> extended_plural_declaration_with_assignment
%type<inf> declaration
%type<inf> assignment
%type<inf> declaration_with_assignment
%type<inf> output










%union
{
  struct info* inf;
};


%%


floating_point: INTEGER '.' INTEGER
                                      {
                                      		STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << "." << $3->str.c_str());
                                      		$$->grid_mapping = GRID_MAPPING_ENTITY;
                                      		$$->array_size = 1;
                                      		$$->type.entity_type =  TYPE_SCALAR;
                                      		$$->type.collection = false;
                                      }
              ;


number: floating_point                { 
											$$ = $1->clone(); 
										}
      | INTEGER
                                      {
                              					$$->str = $1->str.c_str();
                              					$$->grid_mapping = GRID_MAPPING_ENTITY;
                              					$$->array_size = 1;
                              					$$->type.entity_type = TYPE_SCALAR;
                              					$$->type.collection = false;
                              				}
      ;



scalars: expression
                                      {
                                        if($1->error_str.size() == 0)
                                            $$->error_str = $1->error_str;
                                        else
                                          switch($1->type.entity_type)
                                          {
                                              case TYPE_SCALAR:
                                                  if($1->type.collection == false)
                                                  {
                                                      // it should be scalar
                                                      $$->str = $1->str.c_str();
                                                      $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                      $$->array_size = 1;
                                                      $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                      $$->type.collection = false;
                                                  }
                                                  else
                                                  {
                                                      // it should be scalars
                                                      $$->str = $1->str.c_str();
                                                      $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                      $$->array_size = $1->array_size;
                                                      $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                      $$->type.collection = false;
                                                  }
                                                  break;
                                              default:
                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of scalars");
                                                  break;
                                          }
                                      }
       | scalars ',' expression
                                      {
                                        if($3->error_str.size() == 0)
                                            $$->error_str = $3->error_str;
                                        else
                                          switch($3->type.entity_type)
                                          {
                                              case TYPE_SCALAR:
                                                  if($3->type.collection == false)
                                                  {
                                                      // 2nd should be scalar
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str());
                                                      $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                      $$->array_size = $1->array_size + 1;
                                                      $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                      $$->type.collection = false;
                                                  }
                                                  else
                                                  {
                                                      // 2nd should be scalars
                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str());
                                                      $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                      $$->array_size = $1->array_size + $3->array_size;
                                                      $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                      $$->type.collection = false;
                                                  }
                                                  break;
                                              default:
                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of scalar/scalars entities");
                                                  break;
                                          }
                                      }
       ;



vectors: expression
                                          {
                                            if($1->error_str.size() == 0)
                                                $$->error_str = $1->error_str;
                                            else
                                            {
                                                switch($1->type.entity_type)
                                                {
                                                    case TYPE_VECTOR:
                                                        if($1->type.collection == true)
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of vector entities");
                                                        else
                                                        {
                                                            // it should be vector
                                                            $$->str = $1->str.c_str();
                                                            $$->array_size = 1;
                                                            $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                            $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                            $$->type.collection = false;
                                                        }
                                                        break;
                                                    default:
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of vector entities");
                                                        break;
                                                }
                                            }
                                          }
       | vectors ',' expression
                                          {
                                            if($3->error_str.size() == 0)
                                                $$->error_str = $3->error_str;
                                            else
                                            {
                                                switch($3->type.entity_type)
                                                {
                                                    case TYPE_VECTOR:
                                                        if($3->type.collection == true)
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of vector entities");
                                                        else
                                                        {
                                                            // it should be vector
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str());
                                                            $$->array_size = $1->array_size + 1;
                                                            $$->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
                                                            $$->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
                                                            $$->type.collection = false;
                                                        }
                                                        break;
                                                    default:
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The list must contain only of vector entities");
                                                        break;
                                                }
                                            }
                                          }
       ;


expression: '-' expression
                                             {
                                                if($2->error_str.size() == 0)
                                                    $$->error_str = $2->error_str;
                                                else
                                                  switch($2->type.entity_type)
                                                  {
                                                      case TYPE_SCALAR:
														  $$ = $2->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "-" << $2->str.c_str());
                                                          break;
                                                      case TYPE_VECTOR:
														  $$ = $2->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "-" << $2->str.c_str());
                                                          break;
                                                      default:
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Negation not supported for this type");
                                                          break;
                                                  }
                                             }
          | expression '+' expression
                                             {
                                                if($1->error_str.size() == 0)
                                                    $$->error_str = $1->error_str;
                                                else
                                                if($3->error_str.size() == 0)
                                                    $$->error_str = $3->error_str;
                                                else
                                                {
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " + " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == true)
                                                    {  // they both should be scalars
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " + " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be vector

														  $$ = $1->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " + " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == true && $3->type.collection == true)
                                                    {  // both should be vectors
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$= $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " + " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Addition not supported for these types");
                                                    }
                                                }
                                             }
          | expression '-' expression
                                             {
                                                if($1->error_str.size() == 0)
                                                    $$->error_str = $1->error_str;
                                                else
                                                if($3->error_str.size() == 0)
                                                    $$->error_str = $3->error_str;
                                                else
                                                {
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == true)
                                                    {  // they both should be scalars
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$= $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be vector
														$$= $1->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == true && $3->type.collection == true)
                                                    {  // both should be vectors
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Subtraction not supported for these types");
                                                    }
                                                }
                                             }
          | expression '*' expression
                                             {
                                                if($1->error_str.size() == 0)
                                                    $$->error_str = $1->error_str;
                                                else
                                                if($3->error_str.size() == 0)
                                                    $$->error_str = $3->error_str;
                                                else
                                                {
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    { // both should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == true)
                                                    {
                                                        // they both should be scalars
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // 1st should be vector, 2nd should be scalar
														$$ = $1->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // 1st should be scalar, 2nd should be vector
														$$ = $1->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == false)
                                                    {  // 1st should be vectors, 2nd should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == false && $3->type.collection == true)
                                                    {  // 1st should be scalar, 2nd should be vectors
														$$ = $3->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Multiplication not supported for these types");
                                                    }
                                                }
                                             }
          | expression '/' expression
                                             {
                                                if($1->error_str.size() == 0)
                                                    $$->error_str = $1->error_str;
                                                else
                                                if($3->error_str.size() == 0)
                                                    $$->error_str = $3->error_str;
                                                else
                                                {
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == true)
                                                    {  // they both should be scalars
                                                        if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                        {
                                                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                        }
                                                        else
                                                        {
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
                                                        }
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // 1st should be vector, 2nd should be scalar
														$$ = $1->clone();
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_VECTOR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == false)
                                                    {  // 1st should be vectors, 2nd should be scalar
														 $$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " - " << $3->str.c_str());
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Division not supported for these types");
                                                    }
                                                }
                                             }
          | expression '^' expression
                                             {
                                                if($1->error_str.size() == 0)
                                                    $$->error_str = $1->error_str;
                                                else
                                                if($3->error_str.size() == 0)
                                                    $$->error_str = $3->error_str;
                                                else
                                                {
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                    {  // both should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.pow(" << $1->str.c_str() << ", " << $3->str.c_str() << ")");
                                                    }
                                                    else
                                                    if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == false)
                                                    {  // 1st should be scalars, 2nd should be scalar
														$$ = $1->clone();
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.pow(" << $1->str.c_str() << ", " << $3->str.c_str() << ")");
                                                    }
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Power not supported for these types");
                                                    }
                                                }
                                             }
          | number
                                             {
                                                 $$ = $1->clone();
                                             }
          | '(' expression ')'
                                             {
                                                if($2->error_str.size() == 0)
                                                    $$->error_str = $2->error_str;
                                                else
                                                {
                                                    switch($2->type.entity_type)
                                                    {
                                                        case TYPE_SCALAR:
															$$ = $2->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str.c_str() << ")");
                                                            break;
                                                        case TYPE_VECTOR:
															$$ = $2->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "-" << $2->str.c_str());
                                                            break;
                                                        case TYPE_BOOLEAN:
															$$ = $2->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "-" << $2->str.c_str());
                                                            break;
                                                        default:
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Paranthesis embedding not supported for this type");
                                                            break;
                                                    }
                                                }
                                             }
          | EUCLIDEAN_LENGTH '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_VECTOR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.euclideanLength(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.euclideanLength(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Euclidean length not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | LENGTH '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_EDGE:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.length(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.length(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Length not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | AREA '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_FACE:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.area(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.area(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Area not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | VOLUME '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0) {
                                                            $$->error_str = $3->error_str;
														}
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_CELL:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.volume(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.volume(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Volume not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | DOT '(' expression ',' expression ')'
                                                     {
                                                         if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                         else
                                                         if($5->error_str.size() == 0)
                                                            $$->error_str = $5->error_str;
                                                         else
                                                         {
                                                             if($3->type.entity_type == TYPE_VECTOR && $5->type.entity_type == TYPE_VECTOR && $3->type.collection == false && $5->type.collection == false)
                                                             {  // both should be vector
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.dot(" << $3->str.c_str() << ", " << $5->str.c_str() << ")");
                                                                  $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                  $$->array_size = 1;
                                                                  $$->type.entity_type = TYPE_SCALAR;
                                                                  $$->type.collection = false;
                                                             }
                                                             else
                                                             if($3->type.entity_type == TYPE_VECTOR && $5->type.entity_type == TYPE_VECTOR && $3->type.collection == true && $5->type.collection == true)
                                                             {  // they both should be vectors
                                                                 if($3->grid_mapping != $5->grid_mapping || $3->array_size != $5->array_size)    // check that the lengths of the 2 terms are equal
                                                                 {
                                                                    LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                                }
                                                                 else
                                                                 {
                                                                     STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.dot(" << $3->str.c_str() << ", " << $5->str.c_str() << ")");
                                                                     $$->grid_mapping = $3->grid_mapping;
                                                                     $$->array_size = $3->array_size;
                                                                     $$->type.entity_type = TYPE_SCALAR;
                                                                     $$->type.collection = true;
                                                                 }
                                                             }
                                                             else
                                                             {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Dot product not supported for these types");
                                                             }
                                                          }
                                                      }
          | CEIL '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_SCALAR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.ceil(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.ceil(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Ceil not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | FLOOR '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_SCALAR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.floor(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.floor(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Floor not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | ABS '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_SCALAR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.abs(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.abs(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Abs not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | MIN '(' scalars ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.min(" << $3->str.c_str() << ")");
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_SCALAR;
                                                            $$->type.collection = false;
                                                        }
                                                     }
          | MAX '(' scalars ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.max(" << $3->str.c_str() << ")");
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_SCALAR;
                                                            $$->type.collection = false;
                                                        }
                                                     }
          | GRADIENT '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_SCALAR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.gradient(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.gradient(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Gradient not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | DIVERGENCE '(' expression ')'
                                                     {
                                                        if($3->error_str.size() == 0)
                                                            $$->error_str = $3->error_str;
                                                        else
                                                        {
                                                            switch($3->type.entity_type)
                                                            {
                                                                case TYPE_SCALAR:
                                                                    if($3->type.collection == false)
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.divergence(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                        $$->array_size = 1;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = false;
                                                                    }
                                                                    else
                                                                    {
                                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.divergence(" << $3->str.c_str() << ")");
                                                                        $$->grid_mapping = $3->grid_mapping;
                                                                        $$->array_size = $3->array_size;
                                                                        $$->type.entity_type = TYPE_SCALAR;
                                                                        $$->type.collection = true;
                                                                    }
                                                                    break;
                                                                default:
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Divergence not supported for this type");
                                                                    break;
                                                            }
                                                        }
                                                     }
          | '[' scalars ']'
                                                  {
                                                      if($2->error_str.size() == 0)
                                                          $$->error_str = $2->error_str;
                                                      else
                                                      {
                                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "[" << $2->str.c_str() << "]");
                                                          $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                          $$->array_size = $2->array_size;
                                                          $$->type.entity_type = TYPE_VECTOR;
                                                          $$->type.collection = false;
                                                      }
                                                  }
          | CENTROID '(' expression ')'
                                                  {
                                                      if($3->error_str.size() == 0)
                                                          $$->error_str = $3->error_str;
                                                      else
                                                      {
                                                          switch($3->type.entity_type)
                                                          {
                                                              case TYPE_FACE:
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.centroid(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = false;
                                                                  }
                                                                  else
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.centroid(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = $3->grid_mapping;
                                                                      $$->array_size = $3->array_size;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = true;
                                                                  }
                                                                  break;
                                                              case TYPE_CELL:
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.centroid(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = false;
                                                                  }
                                                                  else
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.centroid(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = $3->grid_mapping;
                                                                      $$->array_size = $3->array_size;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = true;
                                                                  }
                                                              default:
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Centroid not supported for this type");
                                                                  break;
                                                          }
                                                      }
                                                  }
          | NORMAL '(' expression ')'
                                                  {
                                                      if($3->error_str.size() == 0)
                                                          $$->error_str = $3->error_str;
                                                      else
                                                      {
                                                          switch($3->type.entity_type)
                                                          {
                                                              case TYPE_FACE:
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.normal(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = false;
                                                                  }
                                                                  else
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.normal(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = $3->grid_mapping;
                                                                      $$->array_size = $3->array_size;
                                                                      $$->type.entity_type = TYPE_VECTOR;
                                                                      $$->type.collection = true;
                                                                  }
                                                                  break;
                                                              default:
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Normal not supported for this type");
                                                                  break;
                                                          }
                                                      }
                                                 }
          | '[' vectors ']'
                                                 {
                                                    if($2->error_str.size() == 0)
                                                        $$->error_str = $2->error_str;
                                                    else
                                                    {
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "[" << $2->str.c_str() << "]");
                                                        $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                        $$->array_size = $2->array_size;
                                                        $$->type.entity_type = TYPE_VECTOR;
                                                        $$->type.collection = true;
                                                    }
                                                 }
          | INTERIOR_VERTICES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorVertices()");
                                                    $$->grid_mapping = GRID_MAPPING_INTERIORVERTICES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_VERTEX;
                                                    $$->type.collection = true;
                                                 }
          | BOUNDARY_VERTICES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryVertices()");
                                                    $$->grid_mapping = GRID_MAPPING_BOUNDARYVERTICES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_VERTEX;
                                                    $$->type.collection = true;
                                                 }
          | ALL_VERTICES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allVertices()");
                                                    $$->grid_mapping = GRID_MAPPING_ALLVERTICES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_VERTEX;
                                                    $$->type.collection = true;
                                                 }
          | INTERIOR_EDGES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorEdges()");
                                                    $$->grid_mapping = GRID_MAPPING_INTERIOREDGES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_EDGE;
                                                    $$->type.collection = true;
                                                 }
          | BOUNDARY_EDGES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryEdges()");
                                                    $$->grid_mapping = GRID_MAPPING_BOUNDARYEDGES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_EDGE;
                                                    $$->type.collection = true;
                                                 }
          | ALL_EDGES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allEdges()");
                                                    $$->grid_mapping = GRID_MAPPING_ALLEDGES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_EDGE;
                                                    $$->type.collection = true;
                                                 }
          | INTERIOR_FACES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorFaces()");
                                                    $$->grid_mapping = GRID_MAPPING_INTERIORFACES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_FACE;
                                                    $$->type.collection = true;
                                                 }
          | BOUNDARY_FACES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryFaces()");
                                                    $$->grid_mapping = GRID_MAPPING_BOUNDARYFACES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_FACE;
                                                    $$->type.collection = true;
                                                 }
          | ALL_FACES '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allFaces()");
                                                    $$->grid_mapping = GRID_MAPPING_ALLFACES;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_FACE;
                                                    $$->type.collection = true;
                                                 }
          | FIRST_CELL '(' expression ')'
                                                 {
                                                      if($3->error_str.size() == 0)
                                                          $$->error_str = $3->error_str;
                                                      else
                                                      {
                                                          switch($3->type.entity_type)
                                                          {
                                                              case TYPE_FACE:
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.firstCell(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_CELL;
                                                                      $$->type.collection = false;
                                                                  }
                                                                  else
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.firstCell(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = $3->grid_mapping;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_CELL;
                                                                      $$->type.collection = true;
                                                                  }
                                                                  break;
                                                              default:
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "FirstCell not supported for this type");
                                                                  break;
                                                          }
                                                      }
                                                 }
          | SECOND_CELL '(' expression ')'
                                                 {
                                                      if($3->error_str.size() == 0)
                                                          $$->error_str = $3->error_str;
                                                      else
                                                      {
                                                          switch($3->type.entity_type)
                                                          {
                                                              case TYPE_FACE:
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.secondCell(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_CELL;
                                                                      $$->type.collection = false;
                                                                  }
                                                                  else
                                                                  {
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.secondCell(" << $3->str.c_str() << ")");
                                                                      $$->grid_mapping = $3->grid_mapping;
                                                                      $$->array_size = 1;
                                                                      $$->type.entity_type = TYPE_CELL;
                                                                      $$->type.collection = true;
                                                                  }
                                                                  break;
                                                              default:
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "SecondCell not supported for this type");
                                                                  break;
                                                          }
                                                      }
                                                 }
          | INTERIOR_CELLS '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorCells()");
                                                    $$->grid_mapping = GRID_MAPPING_INTERIORCELLS;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_CELL;
                                                    $$->type.collection = true;
                                                 }
          | BOUNDARY_CELLS '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryCells()");
                                                    $$->grid_mapping = GRID_MAPPING_BOUNDARYCELLS;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_CELL;
                                                    $$->type.collection = true;
                                                 }
          | ALL_CELLS '(' GRID ')'
                                                 {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allCells()");
                                                    $$->grid_mapping = GRID_MAPPING_ALLCELLS;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_CELL;
                                                    $$->type.collection = true;
                                                 }
          | NOT expression
                                                 {
                                                      if($2->error_str.size() == 0)
                                                          $$->error_str = $2->error_str;
                                                      else
                                                      {
                                                          switch($2->type.entity_type)
                                                          {
                                                              case TYPE_BOOLEAN:
                                                                  if($2->type.collection == false)
                                                                  {
																	  $$ = $2->clone();
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "!" << $2->str.c_str());
                                                                  }
                                                                  else
                                                                  {
																	  $$ = $2->clone();
                                                                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "!" << $2->str.c_str());
                                                                  }
                                                                  break;
                                                              default:
                                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Not not supported for this type");
                                                                  break;
                                                          }
                                                      }
                                                 }
          | expression AND expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be boolean
															$$= $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " && " << $3->str.c_str());
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be booleans
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
																$$ = $1->clone();
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " && " << $3->str.c_str());
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "And not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression OR expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be boolean
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " || " << $3->str.c_str());
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be booleans
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
																$$ = $1->clone();
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " || " << $3->str.c_str());
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Or not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression XOR expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be boolean
															$$ = $1->clone();
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(!" << $1->str.c_str() << " && " << $3->str.c_str() << ") || (!" << $3->str.c_str() << " && " << $1->str.c_str() << ")");
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be booleans
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
																$$ = $1->clone();
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(!" << $1->str.c_str() << " && " << $3->str.c_str() << ") || (!" << $3->str.c_str() << " && " << $1->str.c_str() << ")");
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Xor not supported for these types");
                                                        }
                                                    }
                                                 }
          | TRUE
                                                 {
                                                    $$->str = strdup("true");
                                                    $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_BOOLEAN;
                                                    $$->type.collection = false;
                                                 }
          | FALSE
                                                 {
                                                    $$->str = strdup("false");
                                                    $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                    $$->array_size = 1;
                                                    $$->type.entity_type = TYPE_BOOLEAN;
                                                    $$->type.collection = false;
                                                 }
          | expression '>' expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " > " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") > (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "> not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression '<' expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " < " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") < (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "< not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression LESSEQ expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " <= " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") <= (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "<= not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression GREATEREQ expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " >= " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") >= (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, ">= not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression EQ expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " == " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") == (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "== not supported for these types");
                                                        }
                                                    }
                                                 }
          | expression NOTEQ expression
                                                 {
                                                    if($1->error_str.size() == 0)
                                                        $$->error_str = $1->error_str;
                                                    else
                                                    if($3->error_str.size() == 0)
                                                        $$->error_str = $3->error_str;
                                                    else
                                                    {
                                                        if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == false)
                                                        {
                                                            // both should be scalar
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " != " << $3->str.c_str());
                                                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                                                            $$->array_size = 1;
                                                            $$->type.entity_type = TYPE_BOOLEAN;
                                                            $$->type.collection = false;
                                                        }
                                                        else
                                                        if($1->type.entity_type == TYPE_BOOLEAN && $3->type.entity_type == TYPE_BOOLEAN && $1->type.collection == true && $3->type.collection == true)
                                                        {
                                                            // they should be scalars
                                                            if($1->grid_mapping != $3->grid_mapping)    // check that the lengths of the 2 terms are equal
                                                            {
                                                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                                                            }
                                                            else
                                                            {
                                                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $1->str.c_str() << ") != (" << $3->str.c_str() << ")");
                                                                $$->grid_mapping = $1->grid_mapping;
                                                                $$->array_size = $1->array_size;
                                                                $$->type.entity_type = TYPE_BOOLEAN;
                                                                $$->type.collection = true;
                                                            }
                                                        }
                                                        else
                                                        {
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "!= not supported for these types");
                                                        }
                                                    }
                                                 }
          ;



/*
plural: scalar_exprs            { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | vector_exprs            { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | vertices                { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | edges                   { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | faces                   { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | cells                   { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | adbs                    { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      | boolean_exprs           { $$->str = $1->str.c_str(); $$->grid_mapping = $1->grid_mapping; }
      ;


header: VARIABLE HEADER_DECL SCALAR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Scalar " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL VECTOR                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Vector " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL VERTEX                          { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Vertex " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL EDGE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Edge " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL FACE                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Face " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL CELL                            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Cell " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL ADB                             { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "ScalarAD " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL BOOLEAN                         { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "bool " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF SCALAR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfScalars " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF VECTOR            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfVectors " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF VERTEX            { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfVertices " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF EDGE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfEdges " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF FACE              { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfFaces " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF CELL              { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfCells " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF ADB               { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfScalarsAD " << $1->str.c_str()); }
      | VARIABLE HEADER_DECL COLLECTION OF BOOLEAN           { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfBools " << $1->str.c_str()); }
      ;


parameter_list: header                         { $$->str = $1->str.c_str(); }
              | parameter_list ',' header      { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str()); }
              ;


commands: command1                              { $$->str = $1->str.c_str(); }
        | commands end_lines command1           { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << $2->str.c_str() << $3->str.c_str()); }
        |                                       { $$->str = strdup(""); }     // a function can have only the return instruction
        ;


type: SCALAR                                { $$->str = strdup("Scalar"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | VECTOR                                { $$->str = strdup("Vector"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | VERTEX                                { $$->str = strdup("Vertex"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | EDGE                                  { $$->str = strdup("Edge"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | FACE                                  { $$->str = strdup("Face"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | CELL                                  { $$->str = strdup("Cell"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | ADB                                   { $$->str = strdup("ScalarAD"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | BOOLEAN                               { $$->str = strdup("bool"); $$->grid_mapping = GRID_MAPPING_ENTITY; }
    | COLLECTION OF SCALAR                  { $$->str = strdup("CollOfScalars"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF VECTOR                  { $$->str = strdup("CollOfVectors"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF VERTEX                  { $$->str = strdup("CollOfVertices"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF EDGE                    { $$->str = strdup("CollOfEdges"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF FACE                    { $$->str = strdup("CollOfFaces"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF CELL                    { $$->str = strdup("CollOfCells"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF ADB                     { $$->str = strdup("CollOfScalarsAD"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF BOOLEAN                 { $$->str = strdup("CollOfBools"); $$->grid_mapping = GRID_MAPPING_ANY; }
    | COLLECTION OF SCALAR ON plural        { $$->str = strdup("CollOfScalars"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF VECTOR ON plural        { $$->str = strdup("CollOfVectors"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF VERTEX ON plural        { $$->str = strdup("CollOfVertices"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF EDGE ON plural          { $$->str = strdup("CollOfEdges"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF FACE ON plural          { $$->str = strdup("CollOfFaces"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF CELL ON plural          { $$->str = strdup("CollOfCells"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF ADB ON plural           { $$->str = strdup("CollOfScalarsAD"); $$->grid_mapping = $5->grid_mapping; }
    | COLLECTION OF BOOLEAN ON plural       { $$->str = strdup("CollOfBools"); $$->grid_mapping = $5->grid_mapping; }
    ;
*/

//////////////////////////////////////////////////////////////////////// these support input parameters as expressions with or without ON (option 1)
/*
value: scalar            {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | vector            {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | vertex            {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | edge              {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | face              {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | cell              {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | adb               {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | boolean           {$$->str = $1->str.c_str(); $$->grid_mapping = 1;}
     | scalar_exprs      {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | vector_exprs      {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | vertices          {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | edges             {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | faces             {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | cells             {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | adbs              {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     | booleans          {$$->str = strdup($1->grid_mapping); $$->grid_mapping = $1->grid_mapping;}
     ;


values: value                   {$$->str = $1->str.c_str(); itoa($$->grid_mappings, $1->grid_mappings, 100);}
      | values ',' value        {
                                  char *str = append5($1->str.c_str(),',',$3->str.c_str());
                                  $$->str = strdup(str);
                                  free(str);
                                  char *temp = (char *)malloc(1000 * sizeof(char));
                                  itoa(temp, $3->grid_mapping, 100);
                                  char *str2 = append5($1->grid_mappings,',',temp);
                                  $$->grid_mappings = strdup(str2);
                                  free(str2);
                                }
      ;
*/


//////////////////////////////////////////////////////////////////////// these support input parameters as expressions without ON (option 2)
/*
value: scalar_expr       {$$->str = $1->str.c_str();}
     | vector_expr       {$$->str = $1->str.c_str();}
     | vertex            {$$->str = $1->str.c_str();}
     | edge              {$$->str = $1->str.c_str();}
     | face              {$$->str = $1->str.c_str();}
     | cell              {$$->str = $1->str.c_str();}
     | adb               {$$->str = $1->str.c_str();}
     | boolean_expr      {$$->str = $1->str.c_str();}
     | scalar_exprs      {$$->str = $1->str.c_str();}
     | vector_exprs      {$$->str = $1->str.c_str();}
     | vertices          {$$->str = $1->str.c_str();}
     | edges             {$$->str = $1->str.c_str();}
     | faces             {$$->str = $1->str.c_str();}
     | cells             {$$->str = $1->str.c_str();}
     | adbs              {$$->str = $1->str.c_str();}
     | boolean_exprs     {$$->str = $1->str.c_str();}
     ;


// we need 'values' to be a structure with 2 strings: one which will store the exact output which should be displayed, and another which should store all the terms separated by an unique character ('@')
values: value                   {$$.cCode = $1->str.c_str(); $$.sepCode = $1->str.c_str();}
      | values ',' value        {char *str = append5($1->str.c_str().cCode,',',$3->str.c_str()); $$.cCode = strdup(str); free(str); $$.sepCode = append5($1->str.c_str().sepCode, '@', $3->str.c_str());}
      ;
*/


/*
//////////////////////////////////////////////////////////////////////// this supports input parameters as variables
values: VARIABLE                { $$->str = $1->str.c_str(); }
      | values ',' VARIABLE     { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str()); }
      ;


end_lines: '\n'                 { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "\n"); currentLineNumber++; }
         | '\n' end_lines       { STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "\n" << $2->str.c_str()); currentLineNumber++; }
         |                      { $$->str = strdup(""); }
         ;


return_instr: RETURN boolean_expr '?' VARIABLE ':' VARIABLE
                  {
                    if(check5($4->str.c_str()) == false || check5($6->str.c_str()) == false)
                    {
                        $$->str = strdup("Invalid");
                        $$->grid_mapping = GRID_MAPPING_INVALID;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "return " << $2->str.c_str() << " ? " << $4->str.c_str() << " : " << $6->str.c_str());
                        $$->grid_mapping = getSize3($4->str.c_str());
                    }
                  }

            | RETURN VARIABLE
                  {
                    if(check5($2->str.c_str()) == false)
                    {
                        $$->str = strdup("Invalid");
                        $$->grid_mapping = GRID_MAPPING_INVALID;   // we force it to generate an error message at the function's assignment
                    }
                    else
                    {
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "return " << $2->str.c_str() << ";");
                        $$->grid_mapping = getSize3($2->str.c_str());
                    }
                  }
            ;


function_start: VARIABLE '=' end_lines '{'
                                            {
                                              insideFunction = true;
                                              currentFunctionIndex = getIndex2($1->str.c_str());
                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " = " << $3->str.c_str() << "{");
                                            }


// these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: VARIABLE ':' FUNCTION '(' parameter_list ')' RET type
                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), $1->str.c_str()) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "error at line " << currentLineNumber << ": The function '" << $1->str.c_str() << "' is redeclared");
                                                }
                                                else
                                                {
                                                        fun[funNo++].name = $1->str.c_str();
                                                        fun[funNo-1].type = $8->type;
                                                        fun[funNo-1].grid_mapping = $8->grid_mapping;
                                                        fun[funNo-1].noLocalVariables = 0;
                                                        fun[funNo-1].noParam = 0;

                                                        char *cs1 = $5->str.c_str();    // we need to make a copy, because the strtok function modifies the given string
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
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].type = getVariableType(copy);    // the string we have as a parameter list is already transformed in C++, but we need the types' keywords from Equelle
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].grid_mapping = getGridMapping(copy);  // the string we have as a parameter list is already transformed in C++, but we need the types' lengths
                                                          fun[funNo-1].headerVariables[fun[funNo-1].noParam-1].assigned = true;
                                                          fun[funNo-1].signature = $5->str.c_str();

                                                          pch = strtok(NULL, ",");
                                                        }

                                                        fun[funNo-1].assigned = false;
                                                        // STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $8->str.c_str() << " " << $1->str.c_str() << "(" << $5->str.c_str() << ")" << ";");
                                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "");
                                                }
                                            }
                    ;


function_assignment: function_start end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional

                                            {
                                                int i;
                                                bool declaredBefore = false;

                                                for(i = 0; i < funNo; i++)
                                                    if(strcmp(fun[i].name.c_str(), extract($1->str.c_str())) == 0)
                                                    {
                                                        declaredBefore = true;
                                                        break;
                                                    }

                                                if(declaredBefore == true)
                                                      if(fun[i].assigned == true)
                                                      {
                                                          stringstream ss;
                                                          ss << "error at line " << currentLineNumber << ": The function '" << fun[i].name << "' is reassigned";
                                                          $$->str = strdup(ss.str().c_str());
                                                      }
                                                      else
                                                      {
                                                          if($5->grid_mapping != GRID_MAPPING_INVALID)
                                                          {
                                                              // STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << $2->str.c_str() << $3->str.c_str() << $4->str.c_str() << $5->str.c_str() << $6->str.c_str() << "}");
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "auto " << fun[i].name << "[&](" << fun[i].signature << ") -> " << getStringFromVariableType(fun[i].type) << " {\n" << $2->str.c_str() << $3->str.c_str() << $4->str.c_str() << $5->str.c_str() << $6->str.c_str() << "}");
                                                              if(fun[i].grid_mapping == GRID_MAPPING_ANY && $5->grid_mapping != GRID_MAPPING_ANY)
                                                                  fun[i].grid_mapping = $5->grid_mapping;
                                                              else
                                                                  if(fun[i].grid_mapping != GRID_MAPPING_ANY && $5->grid_mapping == GRID_MAPPING_ANY)
                                                                      {;}   // do nothing (the function must keep its return size from the definition)
                                                                  else
                                                                      {;}   // if both are ANY, the function's return type is already correct; if none are ANY, then they should already be equal, otherwise the instruction flow wouldn't enter on this branch
                                                              fun[i].assigned = true;
                                                          }
                                                          else
                                                          {
                                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "error at line " << currentLineNumber << ": At least one of the return variables does not exist within the function or the return type of the function '" << fun[i].name << "' from its assignment differs than the length of the return type from the function's definition");
                                                          }

                                                      }
                                                else
                                                {
                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "error at line " << currentLineNumber << ": The function '" << extract($1->str.c_str()) <<"' must be declared before being assigned");
                                                }

                                                insideFunction = false;
                                                currentFunctionIndex = -1;
                                            }
                   ;
*/



// function_declaration_with_assignment: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type '=' end_lines '{' end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
//                                     ; // tre sa punem booleana globala true inainte sa execute comenzile din functie



/*
tuple_declaration: VARIABLE ':' TUPLE OF '(' type ')'

tuple_assignment: VARIABLE '=' '(' entities ')'

tuple_declaration_with_assignment: VARIABLE ':' TUPLE OF '(' type ')' '=' '(' entities ')'
*/






output: OUTPUT '(' VARIABLE ')'       { $$->str = output_function($3->str); }








singular_declaration: VARIABLE ':' SCALAR               { $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR, false); }
                    | VARIABLE ':' VECTOR               { $$->str = declaration_function($1->str.c_str(), TYPE_VECTOR, false); }
                    | VARIABLE ':' VERTEX               { $$->str = declaration_function($1->str.c_str(), TYPE_VERTEX, false); }
                    | VARIABLE ':' EDGE                 { $$->str = declaration_function($1->str.c_str(), TYPE_EDGE, false); }
                    | VARIABLE ':' FACE                 { $$->str = declaration_function($1->str.c_str(), TYPE_FACE, false); }
                    | VARIABLE ':' CELL                 { $$->str = declaration_function($1->str.c_str(), TYPE_CELL, false); }
                    | VARIABLE ':' ADB                  { $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR_AD, false); }
                    | VARIABLE ':' BOOLEAN              { $$->str = declaration_function($1->str.c_str(), TYPE_BOOLEAN, false); }
                    ;


plural_declaration: VARIABLE ':' COLLECTION OF SCALAR       { $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR, true); }
                  | VARIABLE ':' COLLECTION OF VECTOR       { $$->str = declaration_function($1->str.c_str(), TYPE_VECTOR, true); }
                  | VARIABLE ':' COLLECTION OF VERTEX       { $$->str = declaration_function($1->str.c_str(), TYPE_VERTEX, true); }
                  | VARIABLE ':' COLLECTION OF EDGE         { $$->str = declaration_function($1->str.c_str(), TYPE_EDGE, true); }
                  | VARIABLE ':' COLLECTION OF FACE         { $$->str = declaration_function($1->str.c_str(), TYPE_FACE, true); }
                  | VARIABLE ':' COLLECTION OF CELL         { $$->str = declaration_function($1->str.c_str(), TYPE_CELL, true); }
                  | VARIABLE ':' COLLECTION OF ADB          { $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR_AD, true); }
                  | VARIABLE ':' COLLECTION OF BOOLEAN      { $$->str = declaration_function($1->str.c_str(), TYPE_BOOLEAN, true); }
                  ;


//TODO: verify that "expression" is a collection
extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_SCALAR, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF VECTOR ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_VECTOR, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF VERTEX ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_VERTEX, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF EDGE ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_EDGE, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF FACE ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_FACE, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF CELL ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_CELL, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF ADB ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_SCALAR_AD, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           | VARIABLE ':' COLLECTION OF BOOLEAN ON expression
                                {
                                  if($7->error_str.size() == 0)
                                      $$->str = $7->error_str;
                                  else
                                  {
                                      if($7->type.collection == false)
                                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration: should be ON collection");
                                      else
                                      {
                                          string out = extended_plural_declaration_function($1->str.c_str(), TYPE_BOOLEAN, $7->str.c_str(), $7->grid_mapping);
                                          $$->str = strdup(out.c_str());
                                      }
                                  }
                                }
                           ;


declaration: singular_declaration           { $$->str = $1->str; }
           | plural_declaration             { $$->str = $1->str; }
           | extended_plural_declaration    { $$->str = $1->str; }
           ;


assignment: VARIABLE '=' USS                      { 
					$$->str = USS_assignment_function($1->str.c_str());
			}
          | VARIABLE '=' USSWD '(' number ')'     { 
					$$->str = USSWD_assignment_function($1->str.c_str(), $5->str.c_str()); 
			}
          | VARIABLE '=' USCOS '(' expression ')'
                                                  {
                                                    if($5->error_str.size() == 0)
                                                        $$->str = $5->error_str;
                                                    else
                                                    {
                                                        if($5->type.collection == false)
                                                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid assignment: USCOS should receive a collection");
                                                        else
                                                        {
                                                            $$->str = USCOS_assignment_function($1->str.c_str(), $5->str.c_str(), $5->grid_mapping);
                                                        }
                                                    }
                                                  }
          | VARIABLE '=' expression
                                              					   {
                        									if($3->error_str.size() == 0)
                                                                  $$->str = $3->error_str;
                                                              else
                                                              {
                                                                  if($3->type.collection == false)
                                                                  {
                                                                      string out = singular_assignment_function($1->str.c_str(), $3);
                                                                      $$->str = strdup(out.c_str());
                                                                  }
                                                                  else
                                                                  {
                                                                      string out = plural_assignment_function($1->str.c_str(), $3);
                                                                      $$->str = strdup(out.c_str());
                                                                  }
                        																		  }
                                              					   }
                   ;


//TODO: verify that "expression" is not a collection
singular_declaration_with_assignment: VARIABLE ':' SCALAR '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' VECTOR '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' VERTEX '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' EDGE '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' FACE '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' CELL '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' ADB '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' BOOLEAN '=' expression
                                                          {
                                                            if($5->error_str.size() == 0)
                                                                $$->str = $5->error_str;
                                                            else
                                                            {
                                                                if($5->type.collection == true)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should not be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $5);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                    | VARIABLE ':' SCALAR '=' USS                   { $$->str = USS_declaration_with_assignment_function($1->str.c_str()); }
                                    | VARIABLE ':' SCALAR '=' USSWD '(' number ')'  { $$->str = USSWD_declaration_with_assignment_function($1->str.c_str(), $7->str.c_str()); }
                                    ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF VECTOR '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF VERTEX '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF EDGE '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF FACE '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF CELL '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF ADB '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF BOOLEAN '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = declaration_with_assignment_function($1->str.c_str(), $7);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  | VARIABLE ':' COLLECTION OF SCALAR '=' USCOS '(' expression ')'
                                                          {
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: USCOS should receive a collection");
                                                                else
                                                                {
                                                                    string out = USCOS_declaration_with_assignment_function($1->str.c_str(), $9->str.c_str(), $9->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                  ;


//TODO: verify that both "expression"s are collections
extended_plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF VECTOR ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF VERTEX ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF EDGE ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF FACE ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF CELL ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF ADB ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF BOOLEAN ON expression '=' expression
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($9->error_str.size() == 0)
                                                                $$->str = $9->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($9->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: the variable should be assigned to a collection");
                                                                else
                                                                {
                                                                    string out = extended_plural_declaration_with_assignment_function($1->str.c_str(), $9, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           | VARIABLE ':' COLLECTION OF SCALAR ON expression '=' USCOS '(' expression ')'
                                                          {
                                                            if($7->error_str.size() == 0)
                                                                $$->str = $7->error_str;
                                                            else
                                                            if($11->error_str.size() == 0)
                                                                $$->str = $11->error_str;
                                                            else
                                                            {
                                                                if($7->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: should be ON collection");
                                                                else
                                                                if($11->type.collection == false)
                                                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Invalid declaration with assignment: USCOS should receive a collection");
                                                                else
                                                                {
                                                                    string out = USCOS_extended_declaration_with_assignment_function($1->str.c_str(), $11->str.c_str(), $7->str.c_str(), $11->grid_mapping, $7->grid_mapping);
                                                                    $$->str = strdup(out.c_str());
                                                                }
                                                            }
                                                          }
                                           ;


 declaration_with_assignment: singular_declaration_with_assignment          { $$->str = $1->str; }
                            | plural_declaration_with_assignment            { $$->str = $1->str; }
                            | extended_plural_declaration_with_assignment   { $$->str = $1->str; }
                            ;




// instructions which can be used in the program and in a function's body
command: declaration                    { $$->str = $1->str; }
       | assignment                     { $$->str = $1->str; }
       | declaration_with_assignment    { $$->str = $1->str; }
       ;

/*
command1: command                       { char* out = $1->str.c_str(); $$->str = out; }
        | command COMMENT               { string st1 = $1->str.c_str(); string st2 = $2->str.c_str(); stringstream ss; ss << st1 << " // " << st2.substr(1, st2.size() - 1); $$->str = strdup(ss.str().c_str()); }
        | COMMENT                       { string st1 = $1->str.c_str(); stringstream ss; ss << "// " << st1.substr(1, st1.size() - 1); $$->str = strdup(ss.str().c_str()); }
        ;
*/

// instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                    { stringstream ss; ss << $1->str.c_str(); $$->str = strdup(ss.str().c_str()); }
   //   | function_declaration                       { stringstream ss; ss << $1->str.c_str(); $$->str = strdup(ss.str().c_str()); }
   //   | function_assignment                        { stringstream ss; ss << $1->str.c_str(); $$->str = strdup(ss.str().c_str()); }
        | output                                     { stringstream ss; ss << $1->str.c_str(); $$->str = strdup(ss.str().c_str()); }
   //   | function_declaration_with_assignment       { stringstream ss; ss << $1->str.c_str(); $$->str = strdup(ss.str().c_str()); }
        ;


pr: pr command2 '\n'                  {
                                        string out = $2->str.c_str();
                                        cout << out << endl;
                                        currentLineNumber++;
                                      }
  | pr command2 COMMENT '\n'          {
                                        string out1 = $2->str.c_str();
                                        string out2 = $3->str.c_str();
                                        cout << out1 << " // " << out2.substr(1, out2.size() - 1) << endl;   //+1 to skip comment sign (#)
                                        currentLineNumber++;
                                      }
  | pr COMMENT '\n'                   {
                                        string out = $2->str.c_str();
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
bool find1(const char* s1, const char* s2)
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
char* find2(const char* s1)
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
char* find3(const char* s1)
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
int find4(const char *s1)
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
char* find5(const char* s1)
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
char* find6(const char* s1)
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
bool check1(const char* s1)
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
bool check2(const char* s1)
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
bool check3(const char* s1)
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
bool check4(const char* s1)
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
bool check5(const char* s1)
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
    if( (fun[currentFunctionIndex].headerVariables[i].type == fun[currentFunctionIndex].type)
		|| (fun[currentFunctionIndex].headerVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping
			&& fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY
			&& fun[currentFunctionIndex].headerVariables[i].grid_mapping != GRID_MAPPING_ANY)
	  )
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
    if((fun[currentFunctionIndex].localVariables[i].type == fun[currentFunctionIndex].type)
		|| (fun[currentFunctionIndex].localVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping
		    && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY
			&& fun[currentFunctionIndex].localVariables[i].grid_mapping != GRID_MAPPING_ANY)
		)
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
bool check6(const char* s1)
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
bool check7(const char* s1)
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
bool check8(const char *s1, const char *s2)
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

      if(var[i].type != getVariableType(pch2))
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
string check9(const char* s1)
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
VariableType getType(const char* variable_name)
{
  HEAP_CHECK();
  int i;
  for(i = 0; i < varNo; i++)
  {
      if(strcmp(variable_name, var[i].name.c_str()) == 0)
      {
        HEAP_CHECK();
        return var[i].type;
      }
  }
  HEAP_CHECK();
  VariableType unknown;
  unknown.entity_type = TYPE_INVALID;
  return unknown;
}


// function which returns the index of a variable, based on its name
int getIndex1(const char* s1)
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
int getIndex2(const char* s1)
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
GridMapping getSize1(const char* s1)
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
GridMapping getSize2(const char* s1)
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
GridMapping getSize3(const char* s1)
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
int getSize4(const char* s1)
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
char* extract(const char* s1)
{
  HEAP_CHECK();
  char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
  char *pch;
  pch = strtok(cs1, " =");
  HEAP_CHECK();
  return pch;
}


// function used to transfer a string within a structure to a separate memory address (of its own)
char *structureToString(const char* st)
{
  HEAP_CHECK();
  char *strA = (char*)malloc(sizeof(char)*strlen(st)+1);
  strcpy(strA, st);
  HEAP_CHECK();
  return strA;
}


// function which converts a type from C++ to its corresponding type in Equelle
VariableType getVariableType(const char* st)
{
	VariableType retval;
	retval.collection = false;
	retval.entity_type = TYPE_INVALID;

    if(strcmp(st, "Scalar") == 0) {
		retval.entity_type = TYPE_SCALAR;
		retval.collection = false;
    }
    else if(strcmp(st, "Vector") == 0) {
      retval.entity_type = TYPE_VECTOR;
	  retval.collection = false;
    }
    else if(strcmp(st, "Vertex") == 0) {
      retval.entity_type =  TYPE_VERTEX;
	  retval.collection = false;
    }
    else if(strcmp(st, "Edge") == 0) {
      retval.entity_type =  TYPE_EDGE;
	  retval.collection = false;
    }
    else if(strcmp(st, "Face") == 0) {
      retval.entity_type =  TYPE_FACE;
	  retval.collection = false;
    }
    else if(strcmp(st, "Cell") == 0) {
      retval.entity_type =  TYPE_CELL;
	  retval.collection = false;
    }
    else if(strcmp(st, "ScalarAD") == 0) {
      retval.entity_type =  TYPE_SCALAR_AD;
	  retval.collection = false;
    }
    else if(strcmp(st, "bool") == 0) {
      retval.entity_type =  TYPE_BOOLEAN;
	  retval.collection = false;
    }
    else if(strcmp(st, "CollOfScalars") == 0) {
      retval.entity_type =  TYPE_SCALAR;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfVectors") == 0) {
      retval.entity_type =  TYPE_VECTOR;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfVertices") == 0) {
      retval.entity_type =  TYPE_VERTEX;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfEdges") == 0) {
      retval.entity_type =  TYPE_EDGE;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfFaces") == 0) {
      retval.entity_type =  TYPE_FACE;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfCells") == 0) {
      retval.entity_type =  TYPE_CELL;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfScalarsAD") == 0) {
      retval.entity_type =  TYPE_SCALAR_AD;
	  retval.collection = true;
    }
    else if(strcmp(st, "CollOfBools") == 0) {
      retval.entity_type =  TYPE_BOOLEAN;
	  retval.collection = true;
    }

	return retval;
}


// function which returns the corresponding size of a C++ type
GridMapping getGridMapping(const char* st)
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


string functionToAnySingularType(const char *st1, const char *st2, const char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
      return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
      return "error2: The function is not assigned";
    }
    if(strcmp(getStringFromVariableType(fun[getIndex2(st1)].type).c_str(), st2) != 0)
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



string functionToAnyCollectionType(const char *st1, const char *st2, const char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
      return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
      return "error2: The function is not assigned";
    }
    if(strcmp(getStringFromVariableType(fun[getIndex2(st1)].type).c_str(), st2) != 0)
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



















std::string declaration_function(const char* variable_name, EntityType entity, bool collection)
{
    HEAP_CHECK();

	string finalString;

    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++) {
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }
		}

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++) {
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                  {
                      taken = true;
                      break;
                  }
			  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.entity_type = entity;
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.collection = collection;
          					if (collection) {
          						fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ANY;
          					}
          					else {
          						fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
          					}
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
              }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            var[varNo++].name = variable_name;
            var[varNo-1].type.entity_type = entity;
            var[varNo-1].type.collection = collection;
      			if (collection) {
      				var[varNo-1].grid_mapping = GRID_MAPPING_ANY;
      			}
      			else {
      				var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
      			}
            var[varNo-1].assigned = false;
        }
    }

    HEAP_CHECK();
    return finalString.c_str();
}






string extended_plural_declaration_function(const char* variable_name, EntityType entity, const char* st3, GridMapping d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                  if(check7(st3) == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << variable_name << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check3(st3) == false)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is undeclared";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(check4(st3) == false)
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is unassigned";
                              finalString = ss.str();
                          }
                          else
                          {
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.entity_type = entity;
                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.collection = true;
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
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
              if(check7(st3) == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << variable_name << "'";
                  finalString = ss.str();
              }
              else
              {
                  if(check1(st3) == false)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is undeclared";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check2(st3) == false)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is unassigned";
                          finalString = ss.str();
                      }
                      else
                      {
                          var[varNo++].name = variable_name;
                          var[varNo-1].type.entity_type = entity;
                          var[varNo-1].grid_mapping = d1;
                          var[varNo-1].assigned = false;
                          var[varNo-1].type.collection = true;
                      }
                  }
              }
        }
    }

    HEAP_CHECK();
    return finalString;
}

/**
  * @param rhs Right hand side of the assignment (a = b+c => rhs is essentially "b+c")
  */
string singular_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
			      ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                {
                    taken = true;
                    break;
                }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local variable '" << variable_name << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check9(rhs->str.c_str()) != "isOk")
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                          finalString = ss.str();
                      }
                      else
                      {
                          if(check6(rhs->str.c_str()) == true)
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(find1(rhs->str.c_str(), variable_name))
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check3(rhs->str.c_str()) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type) << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check4(rhs->str.c_str()) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type) << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(rhs->str.c_str()) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the variable '" << variable_name << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              fun[currentFunctionIndex].localVariables[i].assigned = true;
                                              stringstream ss;
                                              ss << "const " << getStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
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
                  if(check9(rhs->str.c_str()) != "isOk")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check6(rhs->str.c_str()) == true)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(find1(rhs->str.c_str(), variable_name))
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(check3(rhs->str.c_str()) == false)
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check4(rhs->str.c_str()) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check7(rhs->str.c_str()) == true)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          stringstream ss;
                                          ss << "const " << getStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                          finalString = ss.str();
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
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
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(rhs->str.c_str()) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            var[i].assigned = true;
                                            stringstream ss;
                                            ss << "const " << getStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
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
            if(check9(rhs->str.c_str()) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                finalString = ss.str();
            }
            else
            {
                if(check6(rhs->str.c_str()) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(rhs->str.c_str(), variable_name))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(rhs->str.c_str()) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(rhs->str.c_str()) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << getStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = variable_name;
                                    var[varNo-1].type = rhs->type;
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


string plural_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
                  if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The local variable '" << variable_name << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                      finalString = ss.str();
                  }
                  else
                  {
                        if(check9(rhs->str.c_str()) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(rhs->str.c_str()) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(rhs->str.c_str(), variable_name))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(rhs->str.c_str()) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(getSize3(variable_name) != rhs->grid_mapping)
                                                    if(getSize3(variable_name) == GRID_MAPPING_ANY)
                                                    {
                                                        fun[currentFunctionIndex].localVariables[i].grid_mapping = rhs->grid_mapping;
                                                        fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                        stringstream ss;
                                                        ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                        finalString = ss.str();
                                                    }
                                                    else
                                                    {
                                                        stringstream ss;
                                                        ss << "error at line " << currentLineNumber << ": The length of the variable '" << variable_name << "' from its definition differs than the length of its assignment in the function '" << fun[currentFunctionIndex].name << "'";
                                                        finalString = ss.str();
                                                    }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
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
                  if(check9(rhs->str.c_str()) != "isOk")
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                      finalString = ss.str();
                  }
                  else
                  {
                      if(check6(rhs->str.c_str()) == true)
                      {
                          stringstream ss;
                          ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                          finalString = ss.str();
                      }
                      else
                      {
                          if(find1(rhs->str.c_str(), variable_name))
                          {
                              stringstream ss;
                              ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                              finalString = ss.str();
                          }
                          else
                          {
                              if(check3(rhs->str.c_str()) == false)
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check4(rhs->str.c_str()) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check7(rhs->str.c_str()) == true)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          stringstream ss;
                                          ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                          finalString = ss.str();
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                          fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = rhs->grid_mapping;
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
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(rhs->str.c_str()) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(getSize1(variable_name) != rhs->grid_mapping)
                                                if(getSize1(variable_name) == GRID_MAPPING_ANY)
                                                {
                                                    var[i].grid_mapping = rhs->grid_mapping;
                                                    var[i].assigned = true;
                                                    stringstream ss;
                                                    ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
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
            if(check9(rhs->str.c_str()) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                finalString = ss.str();
            }
            else
            {
                if(check6(rhs->str.c_str()) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(rhs->str.c_str(), variable_name))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(rhs->str.c_str()) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(rhs->str.c_str()) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = variable_name;
                                    var[varNo-1].type = rhs->type;
                                    var[varNo-1].grid_mapping = rhs->grid_mapping;
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


string declaration_with_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '%s'" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                              {
                                  stringstream ss;
                                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                  finalString = ss.str();
                              }
                              else
                              {
                                  if(check3(rhs->str.c_str()) == false)
                                  {
                                      stringstream ss;
                                      ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                      finalString = ss.str();
                                  }
                                  else
                                  {
                                      if(check4(rhs->str.c_str()) == false)
                                      {
                                          stringstream ss;
                                          ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                          finalString = ss.str();
                                      }
                                      else
                                      {
                                          if(check7(rhs->str.c_str()) == true)
                                          {
                                              stringstream ss;
                                              ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                              finalString = ss.str();
                                          }
                                          else
                                          {
                                              stringstream ss;
                                              ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                              finalString = ss.str();
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                              fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = rhs->grid_mapping;
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
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(rhs->str.c_str()) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                finalString = ss.str();
            }
            else
            {
                if(check6(rhs->str.c_str()) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(rhs->str.c_str(), variable_name))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(rhs->str.c_str()) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check7(rhs->str.c_str()) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = variable_name;
                                    var[varNo-1].type = rhs->type;
                                    var[varNo-1].grid_mapping = rhs->grid_mapping;
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



string extended_plural_declaration_with_assignment_function(const char* variable_name, const info* rhs, const GridMapping& lhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
              for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                  if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                  {
                      taken = true;
                      break;
                  }

              if(taken == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                  finalString = ss.str();
              }
              else
              {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check3(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check4(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(lhs != rhs->grid_mapping)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The length of The variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            stringstream ss;
                                            ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                            finalString = ss.str();
                                            fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                            fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                            fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = lhs;
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
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
            finalString = ss.str();
        }
        else
        {
            if(check9(rhs->str.c_str()) != "isOk")
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                finalString = ss.str();
            }
            else
            {
                if(check6(rhs->str.c_str()) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                    finalString = ss.str();
                }
                else
                {
                    if(find1(rhs->str.c_str(), variable_name))
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check1(rhs->str.c_str()) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' is undeclared";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check2(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' is unassigned";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(lhs != rhs->grid_mapping)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The length of The variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    stringstream ss;
                                    ss << "const " << getStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                    finalString = ss.str();
                                    var[varNo++].name = variable_name;
                                    var[varNo-1].type = rhs->type;
                                    var[varNo-1].grid_mapping = lhs;
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


















/**
  * USS = User specified scalar
  * a = UserSpecifiedScalar(
  */
string USS_assignment_function(const char* variable_name)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

        if(declaredBefore == true)
              if(var[i].assigned == true)
              {
                  stringstream ss;
                  ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is reassigned";
                  finalString = ss.str();
              }
              else
              {
                  if(var[i].type.entity_type != TYPE_SCALAR && var[i].type.collection != false)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is declared as a " << getStringFromVariableType(var[i].type) << " and cannot be assigned to a scalar";
                      finalString = ss.str();
                  }
                  else
                  {
                      var[i].assigned = true;
                      stringstream ss;
                      ss << "const Scalar " << variable_name << " = param.get<Scalar>(\"" << variable_name << "\");";
                      finalString = ss.str();
                  }
              }
        else
        {
            // deduced declaration
            stringstream ss;
            ss << "const Scalar " << variable_name << " = param.get<Scalar>(\"" << variable_name << "\");";
            finalString = ss.str();
            var[varNo++].name = variable_name;
            var[varNo-1].type.entity_type = TYPE_SCALAR;
			var[varNo-1].type.collection = false;
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USS_declaration_with_assignment_function(const char* st1)
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
            var[varNo-1].type.entity_type = TYPE_SCALAR;
			var[varNo-1].type.collection = false;
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_assignment_function(const char* st1, const char* st2)
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
                  if(var[i].type.entity_type != TYPE_SCALAR && var[i].type.collection != false)
                  {
                      stringstream ss;
                      ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is alreadt declared and cannot be assigned to a scalar";
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
            var[varNo-1].type.entity_type = TYPE_SCALAR;
			var[varNo-1].type.collection = false;
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_declaration_with_assignment_function(const char* st1, const char* st2)
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
            var[varNo-1].type.entity_type = TYPE_SCALAR;
			var[varNo-1].type.collection = false;
            var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
            var[varNo-1].assigned = true;
        }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_assignment_function(const char* st1, const char* st2, GridMapping d1)
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
									var[varNo-1].type.entity_type = TYPE_SCALAR;
									var[varNo-1].type.collection = true;
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


string USCOS_declaration_with_assignment_function(const char* st1, const char* st2, GridMapping d1)
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
									var[varNo-1].type.entity_type = TYPE_SCALAR;
									var[varNo-1].type.collection = false;
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


string USCOS_extended_declaration_with_assignment_function(const char* st1, const char* st2, const char* st3, GridMapping d1, GridMapping d2)
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
												  var[varNo-1].type.entity_type = TYPE_SCALAR;
												  var[varNo-1].type.collection = false;
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


string output_function(std::string& st1)
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
            if(var[i].name == st1)
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


string getStringFromVariableType(VariableType v)
{
	std::stringstream ss;

	switch(v.entity_type) {
		case TYPE_SCALAR:
			ss << ((v.collection) ? "CollOfScalars" : "Scalar");
			break;
		case TYPE_SCALAR_AD:
			ss << ((v.collection) ? "CollOfScalarsAD" : "ScalarAD");
			break;
		case TYPE_VECTOR:
			ss << ((v.collection) ? "CollOfVectors" : "Vector");
			break;
		case TYPE_VERTEX:
			ss << ((v.collection) ? "CollOfVertices" : "Vertex");
			break;
		case TYPE_EDGE:
			ss << ((v.collection) ? "CollOfEdges" : "Edge");
			break;
		case TYPE_FACE:
			ss << ((v.collection) ? "CollOfFaces" : "Face");
			break;
		case TYPE_CELL:
			ss << ((v.collection) ? "CollOfCells" : "Cell");
			break;
		case TYPE_BOOLEAN:
			ss << ((v.collection) ? "CollOfBools" : "bool");
			break;
		case TYPE_INVALID:
			ss << ((v.collection) ? "CollOfInvalidTypes" : "InvalidType");
			break;
		default:
			ss << ((v.collection) ? "CollOfUnknownTypes" : "UnknownType");
	}

	return ss.str();
}
