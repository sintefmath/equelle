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
%type <node> number

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

program: number                 { $$ = new Node(); }
       | COMMENT                { $$ = new Node(); }
       ;

number: INT                     { $$ = createNumber(numFromString(*($1))); delete $1; }
      | FLOAT                   { $$ = createNumber(numFromString(*($1))); delete $1; }
      ;


%%

void yyerror(const char* err)
{
    std::cerr << "Parser error on line " << yylineno << ": " << err << std::endl;
}
