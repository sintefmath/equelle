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
%token BUILTIN
%token ID
%token INT
%token FLOAT
%token COMMENT
%token LEQ
%token GEQ
%token EQ
%token NEQ
%token RET
%token EOL


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
#define YYSTYPE NodePtr
}




%%

program: number                 { $$ = new Node(); }
       | COMMENT                { $$ = new Node(); }
       ;

number: INT                     { $$ = createNumber(numFromString(yytext)); }
      | FLOAT                   { $$ = createNumber(numFromString(yytext)); }
      ;


%%

void yyerror(const char* err)
{
    std::cerr << "Parser error on line " << yylineno << ": " << err << std::endl;
}
