%token COLLECTION
%token OF
%token ON
%token SUBSET
%token SCALAR
%token VECTOR
%token BOOL
%token CELL
%token FACE
%token EDGE
%token VERTEX
%token FUNCTION
%token AND
%token OR
%token NOT
%token XOR
%token TRUE
%token FALSE
%token <str> BUILTIN
%token <str> ID
%token <str> INT
%token <str> FLOAT
%token <str> COMMENT
%token LEQ
%token GEQ
%token EQ
%token NEQ
%token RET
%token EOL

%type <node> program
%type <node> line
%type <node> statement
%type <node> declaration
%type <node> assignment
%type <node> comb_decl_assign
%type <node> expr
%type <node> type_expr
%type <node> basic_type
%type <node> f_decl_args
%type <node> number
%type <node> function_call
%type <node> f_call_args


%output "afryacc.cpp"
%defines "afryacc.hpp"

%start program
%error-verbose


%left OR
%left AND
%nonassoc XOR
%nonassoc EQ NEQ
%nonassoc LEQ GEQ '<' '>'
%left '+' '-'
%left '*'
%nonassoc '/'
%nonassoc '^'
%nonassoc NOT
%left UMINUS



%code requires{
#include "Parser.hpp"
}

%union{
    NodePtr node;
    std::string* str;
}


%%

program: program line           { $$ = new Node(); }
       |                        { $$ = new Node(); }
       ;

line: statement EOL             { $$ = new Node(); }
    | statement COMMENT EOL     { $$ = new Node(); }
    | COMMENT EOL               { $$ = new Node(); }
    | EOL                       { $$ = new Node(); }
    ;

statement: declaration          { $$ = new Node(); }
         | assignment           { $$ = new Node(); }
         | comb_decl_assign     { $$ = new Node(); }
         | expr           { $$ = new Node(); }
;

declaration: ID ':' type_expr  { $$ = new Node(); }

assignment: ID '=' expr   { $$ = new Node(); }

comb_decl_assign: ID ':' type_expr '=' expr  { $$ = new Node(); }

expr: number              { $$ = new Node(); }
    | function_call       { $$ = new Node(); }
    | '(' expr ')'        { $$ = new Node(); }
    | '|' expr '|'        { $$ = new Node(); }
    | expr '/' expr       { $$ = new Node(); }
    | expr '*' expr       { $$ = new Node(); }
    | expr '-' expr       { $$ = new Node(); }
    | '-' expr %prec UMINUS  { $$ = new Node(); }
    | expr '+' expr       { $$ = new Node(); }
    | expr '?' expr ':' expr { $$ = new Node(); }
    | expr ON expr        { $$ = new Node(); }
    | ID                  { $$ = new Node(); }
    ;

type_expr: basic_type     { $$ = new Node(); }
         | COLLECTION OF basic_type     { $$ = new Node(); }
         | COLLECTION OF basic_type ON expr { $$ = new Node(); }
         | COLLECTION OF basic_type SUBSET OF expr { $$ = new Node(); }
         | COLLECTION OF basic_type ON expr SUBSET OF expr { $$ = new Node(); }
         | FUNCTION '(' f_decl_args ')' RET type_expr    { $$ = new Node(); }
         ;

basic_type: SCALAR  { $$ = new Node(); }
          | VECTOR  { $$ = new Node(); }
          | BOOL    { $$ = new Node(); }
          | CELL    { $$ = new Node(); }
          | FACE    { $$ = new Node(); }
          | EDGE    { $$ = new Node(); }
          | VERTEX  { $$ = new Node(); }
          ;

f_decl_args: f_decl_args ',' declaration { $$ = new Node(); }
           | declaration                 { $$ = new Node(); }
           |                             { $$ = new Node(); }
           ;

number: INT                     { $$ = createNumber(numFromString(*($1))); delete $1; }
      | FLOAT                   { $$ = createNumber(numFromString(*($1))); delete $1; }
      ;

function_call: BUILTIN '(' f_call_args ')'  { $$ = new Node(); }
             | ID '(' f_call_args ')'       { $$ = new Node(); }
             ;

f_call_args: f_call_args ',' expr     { $$ = new Node(); }
           | expr                     { $$ = new Node(); }
           |                                { $$ = new Node(); }
           ;

%%

void yyerror(const char* err)
{
    std::cerr << "Parser error on line " << yylineno << ": " << err << std::endl;
}
