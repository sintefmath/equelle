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
#define LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY(dd)           do { stringstream ss; ss << "There is a length mismatch between two terms of an operation"; dd = ss.str(); } while (false)

    //using std::cout;
    //using std::endl;
    //using std::string;
    using namespace std;

} //Code top


%code requires
{
#include "compiler_types.h"

} //Code requires








%code provides
{
#include "parsing_functions.h"
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
%type<inf> scalars
%type<inf> expression
%type<inf> INTEGER
%type<inf> VARIABLE
%type<inf> COMMENT

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
%type<inf> command
%type<inf> command2








%union
{
    struct info* inf;
};


%%
/**
  * Definition of non-terminals
  * -- constructs the parse tree
  */


floating_point: INTEGER '.' INTEGER
{
    $$ = new info();
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
            $$ = new info();
            $$->str = $1->str.c_str();
            $$->grid_mapping = GRID_MAPPING_ENTITY;
            $$->array_size = 1;
            $$->type.entity_type = TYPE_SCALAR;
            $$->type.collection = false;
        }
        ;



scalars: expression
         {
             $$ = new info();
             if($1->error_str.size() > 0)
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
             $$ = new info();
             if($3->error_str.size() > 0)
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


expression: '-' expression
            {
                $$ = new info();
                if($2->error_str.size() > 0)
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                                        if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                                        if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                              if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == false)
                              {   // 1st should be scalars, 2nd should be scalar
                                   $$ = $1->clone();
                                   STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                              }
                              else
                                if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == true)
                                {   // 1st should be scalar, 2nd should be scalars
                                     $$ = $3->clone();
                                     STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
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
                                              STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                          }
                                          else
                                              if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_VECTOR && $1->type.collection == false && $3->type.collection == true)
                                              {  // 1st should be scalar, 2nd should be vectors
                                                  $$ = $3->clone();
                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " * " << $3->str.c_str());
                                              }
                                              else
                                              {
                                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Multiplication not supported for these types");
                                              }
                    }
            }
            | expression '/' expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                              if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == true && $3->type.collection == false)
                              {  // 1st should be scalars, 2nd should be scalar
                                  $$ = $1->clone();
                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
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
                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
                                    }
                                    else
										if($1->type.entity_type == TYPE_SCALAR && $3->type.entity_type == TYPE_SCALAR && $1->type.collection == false && $3->type.collection == true) {
											$$ = $3->clone();
											STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " / " << $3->str.c_str());
										}
										else
										{
											STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Division not supported for these types");
										}
                    }
            }
            | expression '^' expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                $$ = new info();
                if($2->error_str.size() > 0)
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
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str.c_str() << ")");
                        break;
                    case TYPE_BOOLEAN:
                        $$ = $2->clone();
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str.c_str() << ")");
                        break;
                    default:
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Paranthesis embedding not supported for this type");
                        break;
                    }
                }
            }
            | EUCLIDEAN_LENGTH '(' expression ')'
            {
                $$ = new info();
                if($3->error_str.size() > 0)
                    $$->error_str = $3->error_str;
                else
                {
                    switch($3->type.entity_type)
                    {
                    case TYPE_VECTOR:
                        if($3->type.collection == false)
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                            $$->array_size = 1;
                            $$->type.entity_type = TYPE_SCALAR;
                            $$->type.collection = false;
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
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
                $$ = new info();
                if($3->error_str.size() > 0)
                    $$->error_str = $3->error_str;
                else
                {
                    switch($3->type.entity_type)
                    {
                    case TYPE_EDGE:
                        if($3->type.collection == false)
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                            $$->array_size = 1;
                            $$->type.entity_type = TYPE_SCALAR;
                            $$->type.collection = false;
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
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
                $$ = new info();
                if($3->error_str.size() > 0)
                    $$->error_str = $3->error_str;
                else
                {
                    switch($3->type.entity_type)
                    {
                    case TYPE_FACE:
                        if($3->type.collection == false)
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                            $$->array_size = 1;
                            $$->type.entity_type = TYPE_SCALAR;
                            $$->type.collection = false;
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
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
                $$ = new info();
                if($3->error_str.size() > 0)
                {
                    $$->error_str = $3->error_str;
                }
                else
                {
                    switch($3->type.entity_type)
                    {
                    case TYPE_CELL:
                        if($3->type.collection == false)
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
                            $$->grid_mapping = GRID_MAPPING_ENTITY;
                            $$->array_size = 1;
                            $$->type.entity_type = TYPE_SCALAR;
                            $$->type.collection = false;
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.norm(" << $3->str.c_str() << ")");
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
                $$ = new info();
                if($3->error_str.size() > 0)
                    $$->error_str = $3->error_str;
                else
                    if($5->error_str.size() > 0)
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
                                if(($3->grid_mapping != $5->grid_mapping && $3->grid_mapping != GRID_MAPPING_ANY && $5->grid_mapping != GRID_MAPPING_ANY) || $3->array_size != $5->array_size)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                            $$->grid_mapping = GRID_MAPPING_INTERIORFACES;
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                            $$->grid_mapping = GRID_MAPPING_ALLCELLS;
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
                $$ = new info();
                if($2->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                        break;
                    default:
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "Centroid not supported for this type");
                        break;
                    }
                }
            }
            | NORMAL '(' expression ')'
            {
                $$ = new info();
                if($3->error_str.size() > 0)
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
            | INTERIOR_VERTICES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorVertices()");
                $$->grid_mapping = GRID_MAPPING_INTERIORVERTICES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_VERTEX;
                $$->type.collection = true;
            }
            | BOUNDARY_VERTICES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryVertices()");
                $$->grid_mapping = GRID_MAPPING_BOUNDARYVERTICES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_VERTEX;
                $$->type.collection = true;
            }
            | ALL_VERTICES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allVertices()");
                $$->grid_mapping = GRID_MAPPING_ALLVERTICES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_VERTEX;
                $$->type.collection = true;
            }
            | INTERIOR_EDGES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorEdges()");
                $$->grid_mapping = GRID_MAPPING_INTERIOREDGES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_EDGE;
                $$->type.collection = true;
            }
            | BOUNDARY_EDGES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryEdges()");
                $$->grid_mapping = GRID_MAPPING_BOUNDARYEDGES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_EDGE;
                $$->type.collection = true;
            }
            | ALL_EDGES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allEdges()");
                $$->grid_mapping = GRID_MAPPING_ALLEDGES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_EDGE;
                $$->type.collection = true;
            }
            | INTERIOR_FACES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorFaces()");
                $$->grid_mapping = GRID_MAPPING_INTERIORFACES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_FACE;
                $$->type.collection = true;
            }
            | BOUNDARY_FACES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryFaces()");
                $$->grid_mapping = GRID_MAPPING_BOUNDARYFACES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_FACE;
                $$->type.collection = true;
            }
            | ALL_FACES '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allFaces()");
                $$->grid_mapping = GRID_MAPPING_ALLFACES;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_FACE;
                $$->type.collection = true;
            }
            | FIRST_CELL '(' expression ')'
            {
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.interiorCells()");
                $$->grid_mapping = GRID_MAPPING_INTERIORCELLS;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_CELL;
                $$->type.collection = true;
            }
            | BOUNDARY_CELLS '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.boundaryCells()");
                $$->grid_mapping = GRID_MAPPING_BOUNDARYCELLS;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_CELL;
                $$->type.collection = true;
            }
            | ALL_CELLS '(' GRID ')'
            {
                $$ = new info();
                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "er.allCells()");
                $$->grid_mapping = GRID_MAPPING_ALLCELLS;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_CELL;
                $$->type.collection = true;
            }
            | NOT expression
            {
                $$ = new info();
                if($2->error_str.size() > 0)
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
                $$ = new info();
                $$->str = strdup("true");
                $$->grid_mapping = GRID_MAPPING_ENTITY;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_BOOLEAN;
                $$->type.collection = false;
            }
            | FALSE
            {
                $$ = new info();
                $$->str = strdup("false");
                $$->grid_mapping = GRID_MAPPING_ENTITY;
                $$->array_size = 1;
                $$->type.entity_type = TYPE_BOOLEAN;
                $$->type.collection = false;
            }
            | expression '>' expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "> not supported for these types");
                        }
                    }
            }
            | '(' scalars ')' '>' '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") > (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | expression '<' expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "< not supported for these types");
                        }
                    }
            }
            | '(' scalars ')' '<' '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") < (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | expression LESSEQ expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
            | '(' scalars ')' LESSEQ '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") <= (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | expression GREATEREQ expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
            | '(' scalars ')' GREATEREQ '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") >= (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | expression EQ expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
            | '(' scalars ')' EQ '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") == (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | expression NOTEQ expression
            {
                $$ = new info();
                if($1->error_str.size() > 0)
                    $$->error_str = $1->error_str;
                else
                    if($3->error_str.size() > 0)
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
                                if($1->grid_mapping != $3->grid_mapping && $1->grid_mapping != GRID_MAPPING_ANY && $3->grid_mapping != GRID_MAPPING_ANY)    // check that the lengths of the 2 terms are equal
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
            | '(' scalars ')' NOTEQ '(' scalars ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($6->error_str.size() > 0)
                        $$->error_str = $6->error_str;
                    else
                    {
                        if($2->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                        {
                            LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                        }
                        else
                        {
                            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << ") != (" << $6->str << ")");
                            $$->grid_mapping = $2->grid_mapping;
                            $$->array_size = $2->array_size;
                            $$->type.entity_type = TYPE_BOOLEAN;
                            $$->type.collection = true;
                        }
                    }
            }
            | '(' expression '?' expression ':' expression ')'
            {
                $$ = new info();
                if($2->error_str.size() > 0)
                    $$->error_str = $2->error_str;
                else
                    if($4->error_str.size() > 0)
                        $$->error_str = $4->error_str;
                    else
                        if($6->error_str.size() > 0)
                            $$->error_str = $6->error_str;
                        else
                        {
                            if($2->type.entity_type != TYPE_BOOLEAN)
                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The first term of the ternary assignment must be of boolean type");
                            else
                                if($2->type.collection == true)
                                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The first term of the ternary assignment must not be a collection");
                                else
                                    if($4->type != $6->type)
                                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The 2nd and 3rd terms of the ternary assignment must have the same type");
                            if($4->array_size != $6->array_size)    // check that the lengths of the 2 terms are equal
                            {
                                LENGTH_MISMATCH_ERROR_TO_CHAR_ARRAY($$->error_str);
                            }
                            else
                            {
                                STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "(" << $2->str << " ? " << $4->str << " : " << $6->str << ")");
                                $$->grid_mapping = $4->grid_mapping;
                                $$->array_size = $4->array_size;
                                $$->type = $4->type;
                            }
                        }
            }
            | VARIABLE
            {
                $$ = new info();
                int i;
                for(i = 0; i < varNo; i++)
                {
                    if(var[i].name == $1->str)
                        break;
                }
                if(i == varNo)
                {
                    if(insideFunction == false)
                        STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "An undeclared variable is used");
                    else    // it could be a variable from the header or a local variable
                    {
                        bool ok = false;
                        int j;
                        for(j = 0; j < fun[currentFunctionIndex].noParam; j++)
                        {
                            if(fun[currentFunctionIndex].headerVariables[j].name == $1->str)
                            {
                                ok = true;
                                break;
                            }
                        }
                        if(ok == false)   // it could be a local variable
                        {
                            int k;
                            for(k = 0; k < fun[currentFunctionIndex].noLocalVariables; k++)
                            {
                                if(fun[currentFunctionIndex].localVariables[k].name == $1->str)
                                {
                                    ok = true;
                                    break;
                                }
                            }
                            if(ok == true)
                              if(fun[currentFunctionIndex].localVariables[k].assigned == false)
                              {
                                  STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "An unassigned local variable is used");
                              }
                              else
                              {
                                  $$->str = fun[currentFunctionIndex].localVariables[k].name;
                                  $$->grid_mapping = fun[currentFunctionIndex].localVariables[k].grid_mapping;
                                  $$->array_size = fun[currentFunctionIndex].localVariables[k].array_size;
                                  $$->type.entity_type = fun[currentFunctionIndex].localVariables[k].type.entity_type;
                                  $$->type.collection = fun[currentFunctionIndex].localVariables[k].type.collection;
                              }
                            else
                              STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "An undeclared variable is used inside the function");
                        }
                        else
                        {
                            $$->str = fun[currentFunctionIndex].headerVariables[j].name;
                            $$->grid_mapping = fun[currentFunctionIndex].headerVariables[j].grid_mapping;
                            $$->array_size = fun[currentFunctionIndex].headerVariables[j].array_size;
                            $$->type.entity_type = fun[currentFunctionIndex].headerVariables[j].type.entity_type;
                            $$->type.collection = fun[currentFunctionIndex].headerVariables[j].type.collection;
                        }
                    }
                }
                else
                {
                    $$->str = var[i].name;
                    $$->grid_mapping = var[i].grid_mapping;
                    $$->array_size = var[i].array_size;
                    $$->type.entity_type = var[i].type.entity_type;
                    $$->type.collection = var[i].type.collection;
                }
            }
            | VARIABLE '(' values ')'              {
                $$ = new info();
                int i;
                for(i = 0; i < funNo; i++)
                {
                    if(fun[i].name == $1->str)
                        break;
                }
                if(i == funNo) {
                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "An undeclared function is called");
                }
                else
                {
                    string st;
                    if(fun[i].type.collection == false)
                    {
                        st = functionToAnySingularType($1->str.c_str(), getCppTypeStringFromVariableType(fun[i].type).c_str(), $3->str.c_str(), getEquelleTypeStringFromVariableType(fun[i].type));
                    }
                    else
                    {
                        st = functionToAnyCollectionType($1->str.c_str(), getCppTypeStringFromVariableType(fun[i].type).c_str(), $3->str.c_str(), getEquelleTypeStringFromVariableType(fun[i].type));
                    }

                    if(check9(st.c_str()) != "isOk") {
                        $$->error_str = st;
					}
                    else
                    {
						if (st == "ok") {
							STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str << "(" << $3->str << ")");
							$$->grid_mapping = fun[i].grid_mapping;
							$$->type.entity_type = fun[i].type.entity_type;
							$$->type.collection = fun[i].type.collection;
						}
						else {
							$$->error_str = st;
						}
                    }

                }
            }
            ;


header: VARIABLE HEADER_DECL SCALAR                          {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Scalar " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL VECTOR                          {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Vector " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL VERTEX                          {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Vertex " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL EDGE                            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Edge " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL FACE                            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Face " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL CELL                            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "Cell " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL ADB                             {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "ScalarAD " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL BOOLEAN                         {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "bool " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF SCALAR            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfScalars " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF VECTOR            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfVectors " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF VERTEX            {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfVertices " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF EDGE              {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfEdges " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF FACE              {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfFaces " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF CELL              {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfCells " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF ADB               {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfScalarsAD " << $1->str.c_str());
        }
        | VARIABLE HEADER_DECL COLLECTION OF BOOLEAN           {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "CollOfBools " << $1->str.c_str());
        }
        ;


parameter_list: header                         {
        $$ = new info();
        $$->str = $1->str;
                }
                | parameter_list ',' header      {
                    $$ = new info();
                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str << ", " << $3->str);
                }
                ;


commands: command1                              {
                $$ = new info();
                $$->str = $1->str.c_str();
          }
          | commands end_lines command1           {
              $$ = new info();
              STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << $2->str.c_str() << $3->str.c_str());
          }
          |                                       {
              $$ = new info();
              $$->str = strdup("");
          }     // a function can have only the return instruction
          ;


type: SCALAR                                {
          $$ = new info();
          $$->str = strdup("Scalar");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_SCALAR;
          $$->type.collection = false;
      }
      | VECTOR                                {
          $$ = new info();
          $$->str = strdup("Vector");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_VECTOR;
          $$->type.collection = false;

      }
      | VERTEX                                {
          $$ = new info();
          $$->str = strdup("Vertex");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_VERTEX;
          $$->type.collection = false;
      }
      | EDGE                                  {
          $$ = new info();
          $$->str = strdup("Edge");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_EDGE;
          $$->type.collection = false;
      }
      | FACE                                  {
          $$ = new info();
          $$->str = strdup("Face");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_FACE;
          $$->type.collection = false;
      }
      | CELL                                  {
          $$ = new info();
          $$->str = strdup("Cell");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_CELL;
          $$->type.collection = false;
      }
      | ADB                                   {
          $$ = new info();
          $$->str = strdup("ScalarAD");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_SCALAR_AD;
          $$->type.collection = false;
      }
      | BOOLEAN                               {
          $$ = new info();
          $$->str = strdup("bool");
          $$->grid_mapping = GRID_MAPPING_ENTITY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_BOOLEAN;
          $$->type.collection = false;
      }
      | COLLECTION OF SCALAR                  {
          $$ = new info();
          $$->str = strdup("CollOfScalars");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_SCALAR;
          $$->type.collection = true;
      }
      | COLLECTION OF VECTOR                  {
          $$ = new info();
          $$->str = strdup("CollOfVectors");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_VECTOR;
          $$->type.collection = true;
      }
      | COLLECTION OF VERTEX                  {
          $$ = new info();
          $$->str = strdup("CollOfVertices");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_VERTEX;
          $$->type.collection = true;
      }
      | COLLECTION OF EDGE                    {
          $$ = new info();
          $$->str = strdup("CollOfEdges");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_EDGE;
          $$->type.collection = true;
      }
      | COLLECTION OF FACE                    {
          $$ = new info();
          $$->str = strdup("CollOfFaces");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_FACE;
          $$->type.collection = true;
      }
      | COLLECTION OF CELL                    {
          $$ = new info();
          $$->str = strdup("CollOfCells");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_CELL;
          $$->type.collection = true;
      }
      | COLLECTION OF ADB                     {
          $$ = new info();
          $$->str = strdup("CollOfScalarsAD");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_SCALAR_AD;
          $$->type.collection = true;
      }
      | COLLECTION OF BOOLEAN                 {
          $$ = new info();
          $$->str = strdup("CollOfBools");
          $$->grid_mapping = GRID_MAPPING_ANY;
          $$->array_size = 1;
          $$->type.entity_type = TYPE_BOOLEAN;
          $$->type.collection = true;
      }
      | COLLECTION OF SCALAR ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfScalars");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_SCALAR;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF VECTOR ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfVectors");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_VECTOR;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF VERTEX ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfVertices");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_VERTEX;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF EDGE ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfEdges");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_EDGE;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF FACE ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfFaces");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_FACE;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF CELL ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfCells");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_CELL;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF ADB ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfScalarsAD");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_SCALAR_AD;
              $$->type.collection = true;
          }
      }
      | COLLECTION OF BOOLEAN ON expression
      {
          $$ = new info();
          if($5->error_str.size() > 0)
              $$->error_str = $5->error_str;
          else
          {
              $$->str = strdup("CollOfBools");
              $$->grid_mapping = $5->grid_mapping;
              $$->array_size = $5->array_size;
              $$->type.entity_type = TYPE_BOOLEAN;
              $$->type.collection = true;
          }
      }
      ;


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



      //////////////////////////////////////////////////////////////////////// this supports input parameters as variables
values: VARIABLE                {
      $$ = new info();
      $$->str = $1->str.c_str();
        }
        | values ',' VARIABLE     {
            $$ = new info();
            STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << ", " << $3->str.c_str());
        }
        ;


end_lines: '\n'                 {
        $$ = new info();
        STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "\n"); currentLineNumber++;
           }
           | '\n' end_lines       {
               $$ = new info();
               STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "\n" << $2->str.c_str()); currentLineNumber++;
           }
           |                      {
               $$ = new info();
               $$->str = strdup("");
           }
           ;


return_instr: RETURN expression '?' VARIABLE ':' VARIABLE       // TODO: check that expression is boolean type
              {
                  $$ = new info();
                  if($2->type.entity_type != TYPE_BOOLEAN || $2->type.collection == false)
                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The first term of the ternary expression must be a boolean type and not a collection");
                  else
                      if(check5($4->str.c_str()) == false || check5($6->str.c_str()) == false)
                      {
                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "One of the return variables from the ternary expression does not meet the requirements");
                      }
                      else
                      {
                          STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "return " << $2->str.c_str() << " ? " << $4->str.c_str() << " : " << $6->str.c_str());
                          $$->grid_mapping = getSize3($4->str.c_str());
                      }
              }

              | RETURN VARIABLE
              {
                  $$ = new info();
                  if(check5($2->str.c_str()) == false)
                  {
                      STREAM_TO_DOLLARS_CHAR_ARRAY($$->error_str, "The return variable from the ternary expression does not meet the requirements");
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
                    $$ = new info();
                    insideFunction = true;
                    currentFunctionIndex = getIndex2($1->str.c_str());
                    STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, $1->str.c_str() << " = " << $3->str.c_str() << "{");
                }


                // these 3 instruction types must not be part of the body of another function ==> we need to separate the commands which can be used inside a function's body from the commands which can be used in the program
function_declaration: VARIABLE ':' FUNCTION '(' parameter_list ')' RET type
                      {
                          $$ = new info();
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
                                  fun[funNo++].name = $1->str;
                                  fun[funNo-1].type = $8->type;
                                  fun[funNo-1].grid_mapping = $8->grid_mapping;
                                  fun[funNo-1].noLocalVariables = 0;
                                  fun[funNo-1].noParam = 0;

                                  char *cs1 = strdup($5->str.c_str());    // we need to make a copy, because the strtok function modifies the given string
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
                                      fun[funNo-1].signature = $5->str;

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
                         $$ = new info();
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
                                         STREAM_TO_DOLLARS_CHAR_ARRAY($$->str, "auto " << fun[i].name << " = [&](" << fun[i].signature << ") -> " << getCppTypeStringFromVariableType(fun[i].type) << " {\n" << $2->str.c_str() << $3->str.c_str() << $4->str.c_str() << $5->str.c_str() << $6->str.c_str() << "}");
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




                     // function_declaration_with_assignment: FUNCTION_VARIABLE ':' FUNCTION '(' parameter_list ')' "->" type '=' end_lines '{' end_lines commands end_lines return_instr end_lines '}'    // the end lines are optional
                     //                                     ; // we must set the global bool variable "insideFunction" to be true before the commands inside the function are parsed



                     /*
                     tuple_declaration: VARIABLE ':' TUPLE OF '(' type ')'

                     tuple_assignment: VARIABLE '=' '(' entities ')'

                     tuple_declaration_with_assignment: VARIABLE ':' TUPLE OF '(' type ')' '=' '(' entities ')'
                     */






output: OUTPUT '(' VARIABLE ')'       { $$ = new info();
                     $$->str = output_function($3->str);
        }








singular_declaration: VARIABLE ':' SCALAR               {
        $$ = new info();
        $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR, false);
                      }
                      | VARIABLE ':' VECTOR               {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_VECTOR, false);
                      }
                      | VARIABLE ':' VERTEX               {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_VERTEX, false);
                      }
                      | VARIABLE ':' EDGE                 {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_EDGE, false);
                      }
                      | VARIABLE ':' FACE                 {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_FACE, false);
                      }
                      | VARIABLE ':' CELL                 {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_CELL, false);
                      }
                      | VARIABLE ':' ADB                  {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR_AD, false);
                      }
                      | VARIABLE ':' BOOLEAN              {
                          $$ = new info();
                          $$->str = declaration_function($1->str.c_str(), TYPE_BOOLEAN, false);
                      }
                      ;


plural_declaration: VARIABLE ':' COLLECTION OF SCALAR       {
                      $$ = new info();
                      $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR, true);
                    }
                    | VARIABLE ':' COLLECTION OF VECTOR       {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_VECTOR, true);
                    }
                    | VARIABLE ':' COLLECTION OF VERTEX       {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_VERTEX, true);
                    }
                    | VARIABLE ':' COLLECTION OF EDGE         {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_EDGE, true);
                    }
                    | VARIABLE ':' COLLECTION OF FACE         {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_FACE, true);
                    }
                    | VARIABLE ':' COLLECTION OF CELL         {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_CELL, true);
                    }
                    | VARIABLE ':' COLLECTION OF ADB          {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_SCALAR_AD, true);
                    }
                    | VARIABLE ':' COLLECTION OF BOOLEAN      {
                        $$ = new info();
                        $$->str = declaration_function($1->str.c_str(), TYPE_BOOLEAN, true);
                    }
                    ;


                    //TODO: verify that "expression" is a collection
extended_plural_declaration: VARIABLE ':' COLLECTION OF SCALAR ON expression
                             {
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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
                                 $$ = new info();
                                 if($7->error_str.size() > 0)
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


declaration: singular_declaration           {
                             $$ = new info();
                             $$->str = $1->str;
             }
             | plural_declaration             {
                 $$ = new info();
                 $$->str = $1->str;
             }
             | extended_plural_declaration    {
                 $$ = new info();
                 $$->str = $1->str;
             }
             ;


assignment: VARIABLE '=' USS
            {
                $$ = new info();
                $$->str = USS_assignment_function($1->str.c_str());
            }
            | VARIABLE '=' USSWD '(' number ')'
            {
                $$ = new info();
                $$->str = USSWD_assignment_function($1->str.c_str(), $5->str.c_str());
            }
            | VARIABLE '=' USCOS '(' expression ')'
            {
                $$ = new info();
                if($5->error_str.size() > 0)
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
                $$ = new info();
                if($3->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                          $$ = new info();
                                          if($5->error_str.size() > 0)
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
                                      | VARIABLE ':' SCALAR '=' USS                   {
                                          $$ = new info();
                                          $$->str = USS_declaration_with_assignment_function($1->str.c_str());
                                      }
                                      | VARIABLE ':' SCALAR '=' USSWD '(' number ')'  {
                                          $$ = new info();
                                          $$->str = USSWD_declaration_with_assignment_function($1->str.c_str(), $7->str.c_str());
                                      }
                                      ;


plural_declaration_with_assignment: VARIABLE ':' COLLECTION OF SCALAR '=' expression
                                    {
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($7->error_str.size() > 0)
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
                                        $$ = new info();
                                        if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($9->error_str.size() > 0)
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
                                                 $$ = new info();
                                                 if($7->error_str.size() > 0)
                                                     $$->str = $7->error_str;
                                                 else
                                                     if($11->error_str.size() > 0)
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


declaration_with_assignment: singular_declaration_with_assignment          {
                                             $$ = new info();$$->str = $1->str; }
                             | plural_declaration_with_assignment            {
                                 $$ = new info();$$->str = $1->str; }
                             | extended_plural_declaration_with_assignment   {
                                 $$ = new info();$$->str = $1->str; }
                             ;




                             // instructions which can be used in the program and in a function's body
command: declaration                    {
                             $$ = new info();$$->str = $1->str; }
         | assignment                     {
             $$ = new info();$$->str = $1->str; }
         | declaration_with_assignment    {
             $$ = new info();$$->str = $1->str; }
         ;


command1: command                       {
         $$ = new info();$$->str = $1->str; }
          | command COMMENT               {
              $$ = new info();string st1 = $1->str.c_str(); string st2 = $2->str.c_str(); stringstream ss; ss << st1 << " // " << st2.substr(1, st2.size() - 1); $$->str = strdup(ss.str().c_str()); }
          | COMMENT                       {
              $$ = new info();string st1 = $1->str.c_str(); stringstream ss; ss << "// " << st1.substr(1, st1.size() - 1); $$->str = strdup(ss.str().c_str()); }
          ;


          // instructions which can be used in the program, but not in a function's body (since we must not allow inner functions)
command2: command                                      { $$ = new info(); $$->str = $1->str; }
          | function_declaration                       { $$ = new info(); $$->str = $1->str; }
          | function_assignment                        { $$ = new info(); $$->str = duplicateFunction($1->str); }
          | output                                     { $$ = new info(); $$->str = $1->str; }
      //  | function_declaration_with_assignment       { $$ = new info(); $$->str = $1->str; }
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
/**
  * C++ part, which contains function implementations as used above
  */
extern int yylex();
extern int yyparse();

#include "parsing_functions.cpp"
#include "compiler_types.cpp"