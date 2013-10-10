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

%type <seq> program
%type <node> line
%type <node> statement
%type <vardecl> declaration
%type <node> f_declaration
%type <node> assignment
%type <node> comb_decl_assign
%type <node> expr
%type <type> type_expr
%type <ftype> f_type_expr
%type <type> basic_type
%type <fargdecl> f_decl_args
%type <node> number
%type <node> function_call
%type <farg> f_call_args
%type <node> f_body
%type <node> f_startdef


%output "equelle_parser.cpp"
%defines "equelle_parser.hpp"

%start program
%error-verbose

%nonassoc '?'
%nonassoc ON
%left OR
%nonassoc XOR
%left AND
%nonassoc EQ NEQ
%nonassoc LEQ GEQ '<' '>'
%left '+' '-'
%left '*'
%nonassoc '/'
%nonassoc '^'
%nonassoc NOT UMINUS



%code requires{
#include "ParseActions.hpp"
#include <iostream>
}

%union{
    Node* node;
    TypeNode* type;
    VarDeclNode* vardecl;
    FuncTypeNode* ftype;
    FuncArgsNode* farg;
    FuncArgsDeclNode* fargdecl;
    SequenceNode* seq;
    std::string* str;
}


%%

program: program line           { $$ = $1; $$->pushNode($2); }
       |                        { $$ = new SequenceNode(); }
       ;

line: statement EOL             { $$ = $1; }
    | statement COMMENT EOL     { $$ = $1; }
    | COMMENT EOL               { $$ = 0; }
    | EOL                       { $$ = 0; }
    ;

f_body: '{' EOL program '}'     { $$ = handleFuncBody($3); }

statement: declaration          { $$ = $1; }
         | f_declaration        { $$ = $1; }
         | assignment           { $$ = $1; }
         | comb_decl_assign     { $$ = $1; }
         | function_call        { $$ = $1; }
         | RET expr             { $$ = handleReturnStatement($2); }
         ;

declaration: ID ':' type_expr  { $$ = handleDeclaration(*($1), $3); delete $1; }

f_declaration: ID ':' f_type_expr  { $$ = handleFuncDeclaration(*($1), $3); delete $1; }

assignment: ID '=' expr   { $$ = handleAssignment(*($1), $3); delete $1; }
          | f_startdef f_body  { $$ = new FuncAssignNode($1, $2); }
          ;

f_startdef: ID '(' f_call_args ')' '='       { $$ = handleFuncStart(*($1), $3); delete $1; }

comb_decl_assign: ID ':' type_expr '=' expr  { $$ = handleDeclarationAssign(*($1), $3, $5); delete $1; }

expr: number              { $$ = $1; }
    | function_call       { $$ = $1; }
    | '(' expr ')'        { $$ = $2; }
    | '|' expr '|'        { $$ = handleNorm($2); }
    | expr '/' expr       { $$ = handleBinaryOp(Divide, $1, $3); }
    | expr '*' expr       { $$ = handleBinaryOp(Multiply, $1, $3); }
    | expr '-' expr       { $$ = handleBinaryOp(Subtract, $1, $3); }
    | expr '+' expr       { $$ = handleBinaryOp(Add, $1, $3); }
    | '-' expr %prec UMINUS  { $$ = new UnaryNegationNode($2); }
    | expr '?' expr ':' expr %prec '?' { $$ = new TrinaryIfNode($1, $3, $5); }
    | expr ON expr        { $$ = new OnNode($1, $3); }
    | ID                  { $$ = new VarNode(*($1)); delete $1; }
    ;

type_expr: basic_type                                      { $$ = $1; }
         | COLLECTION OF basic_type                        { $$ = handleCollection($3,  0,  0); }
         | COLLECTION OF basic_type ON expr                { $$ = handleCollection($3, $5,  0); }
         | COLLECTION OF basic_type SUBSET OF expr         { $$ = handleCollection($3,  0, $6); }
         ;

f_type_expr: FUNCTION '(' f_decl_args ')' RET type_expr      { $$ = handleFuncType($3, $6); }

basic_type: SCALAR  { $$ = new TypeNode(EquelleType(Scalar)); }
          | VECTOR  { $$ = new TypeNode(EquelleType(Vector)); }
          | BOOL    { $$ = new TypeNode(EquelleType(Bool)); }
          | CELL    { $$ = new TypeNode(EquelleType(Cell)); }
          | FACE    { $$ = new TypeNode(EquelleType(Face)); }
          | EDGE    { $$ = new TypeNode(EquelleType(Edge)); }
          | VERTEX  { $$ = new TypeNode(EquelleType(Vertex)); }
          ;

f_decl_args: f_decl_args ',' declaration { $$ = $1; $$->addArg($3); }
           | declaration                 { $$ = new FuncArgsDeclNode($1); }
           |                             { $$ = new FuncArgsDeclNode(); }
           ;

number: INT                     { $$ = handleNumber(numFromString(*($1))); delete $1; }
      | FLOAT                   { $$ = handleNumber(numFromString(*($1))); delete $1; }
      ;

function_call: BUILTIN '(' f_call_args ')'  { $$ = handleFuncCall(*($1), $3); delete $1; }
             | ID '(' f_call_args ')'       { $$ = handleFuncCall(*($1), $3); delete $1; }
             ;

f_call_args: f_call_args ',' expr     { $$ = $1; $$->addArg($3); }
           | expr                     { $$ = new FuncArgsNode($1); }
           |                          { $$ = new FuncArgsNode(); }
           ;

%%

void yyerror(const char* err)
{
    std::cerr << "Parser error near line " << yylineno << ": " << err << std::endl;
}
