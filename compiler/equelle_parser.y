%{
/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/
%}

%glr-parser

%token STENCIL
%token COLLECTION
%token SEQUENCE
%token ARRAY
%token OF
%token ON
%token EXTEND
%token SUBSET
%token SCALAR
%token VECTOR
%token BOOL
%token CELL
%token FACE
%token EDGE
%token VERTEX
%token STRING
%token FUNCTION
%token AND
%token OR
%token NOT
%token XOR
%token TRUE
%token FALSE
%token FOR
%token IN
%token MUTABLE
%token DOUBLELB
%token DOUBLERB
%token <str> BUILTIN
%token <str> ID
%token <str> INT
%token <str> FLOAT
%token <str> COMMENT
%token <str> STRING_LITERAL
%token LEQ
%token GEQ
%token EQ
%token NEQ
%token RET
%token EOL
%token LINECONT

%type <seq> program
%type <seq> lineblock
%type <node> line
%type <node> statement
%type <vardecl> declaration
%type <node> f_declaration
%type <node> assignment
%type <node> comb_decl_assign
%type <enode> expr
%type <type> type_expr
%type <type> collection_of
%type <ftype> f_type_expr
%type <type> basic_type
%type <fargdecl> f_decl_args
%type <numnode> number
%type <enode> quantity
%type <unitnode> unit_expr
%type <enode> array
%type <fsnode> f_assign_start
%type <fcalllike> f_call_like
%type <farg> f_call_args
%type <seq> block
%type <loop> loop_start

%start program
%error-verbose

%nonassoc LOWEST
%nonassoc MUTABLE
%nonassoc STENCIL
%nonassoc '?'
%nonassoc ON
%nonassoc EXTEND
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
%left '['
%nonassoc UNIT



%code requires{
#include "ParseActions.hpp"
#include <iostream>
}

%union{
    Node*                          node;
    ExpressionNode*                enode;
    TypeNode*                      type;
    VarDeclNode*                   vardecl;
    FuncTypeNode*                  ftype;
    FuncArgsNode*                  farg;
    FuncStartNode*                 fsnode;
    FuncArgsDeclNode*              fargdecl;
    FuncCallLikeNode*              fcalllike;
    SequenceNode*                  seq;
    NumberNode*                    numnode;
    UnitNode*                      unitnode;
    LoopNode*                      loop;
    std::string*                   str;
}


%%

program: lineblock                { $$ = handleProgram($1); }

lineblock: lineblock line         { $$ = $1; $$->pushNode($2); }
         |                        { $$ = new SequenceNode(); }
         ;

line: statement EOL             { $$ = $1; }
    | statement COMMENT EOL     { $$ = $1; }
    | COMMENT EOL               { $$ = nullptr; }
    | EOL                       { $$ = nullptr; }
    ;

block: '{' EOL lineblock '}'     { $$ = $3; }
     | EOL '{' EOL lineblock '}' { $$ = $4; }

statement: declaration          { $$ = $1; }
         | f_declaration        { $$ = $1; }
         | assignment           { $$ = $1; }
         | comb_decl_assign     { $$ = $1; }
         | f_call_like          { $$ = handleFuncCallStatement($1); }
         | RET expr             { $$ = handleReturnStatement($2); }
         | loop_start block     { $$ = handleLoopStatement($1, $2); }
         ;

declaration: ID ':' type_expr  { $$ = handleDeclaration(*($1), $3); delete $1; }

f_declaration: ID ':' f_type_expr  { $$ = handleFuncDeclaration(*($1), $3); delete $1; }
			
assignment: ID '=' expr       { $$ = handleAssignment(*($1), $3); delete $1; }
          | f_assign_start expr { $$ = handleStencilAssignment($1, $2); }
          | f_assign_start block  { $$ = handleFuncAssignment($1, $2); }
          ;

comb_decl_assign: ID ':' type_expr '=' expr  { $$ = handleDeclarationAssign(*($1), $3, $5); delete $1; }

expr: quantity            { $$ = $1; }
    | f_call_like         { $$ = $1; }
    | expr '[' INT ']'    { $$ = handleRandomAccess($1, intFromString(*($3))); delete $3; }
    | '(' expr ')'        { $$ = $2; }
    | '|' expr '|'        { $$ = handleNorm($2); }
    | expr '/' expr       { $$ = handleBinaryOp(Divide, $1, $3); }
    | expr '*' expr       { $$ = handleBinaryOp(Multiply, $1, $3); }
    | expr '-' expr       { $$ = handleBinaryOp(Subtract, $1, $3); }
    | expr '+' expr       { $$ = handleBinaryOp(Add, $1, $3); }
    | expr '<' expr       { $$ = handleComparison(Less, $1, $3); }
    | expr '>' expr       { $$ = handleComparison(Greater, $1, $3); }
    | expr LEQ expr       { $$ = handleComparison(LessEqual, $1, $3); }
    | expr GEQ expr       { $$ = handleComparison(GreaterEqual, $1, $3); }
    | expr EQ expr        { $$ = handleComparison(Equal, $1, $3); }
    | expr NEQ expr       { $$ = handleComparison(NotEqual, $1, $3); }
    | '-' expr %prec UMINUS  { $$ = handleUnaryNegation($2); }
    | expr '?' expr ':' expr %prec '?' { $$ = handleTrinaryIf($1, $3, $5); }
    | expr ON expr        { $$ = handleOn($1, $3); }
    | expr EXTEND expr    { $$ = handleExtend($1, $3); }
    | ID                  { $$ = handleIdentifier(*($1)); delete $1; }
    | STRING_LITERAL      { $$ = handleString(*($1)); delete $1; }
    | array               { $$ = $1; }
    ;

type_expr: basic_type                                  { $$ = $1; }
		 | collection_of                               { $$ = $1; }
		 | STENCIL collection_of                       { $$ = handleStencilCollection($2); }
         | SEQUENCE OF basic_type                      { $$ = handleSequence($3); }
         | ARRAY OF INT type_expr                      { $$ = handleArrayType(intFromString(*($3)), $4); delete $3; }
         | MUTABLE type_expr                           { $$ = handleMutableType($2); }
         ;
         
collection_of: COLLECTION OF basic_type                    { $$ = handleCollection($3, nullptr,  nullptr); }
             | COLLECTION OF basic_type ON expr            { $$ = handleCollection($3,      $5,  nullptr); }
             | COLLECTION OF basic_type SUBSET OF expr     { $$ = handleCollection($3, nullptr,       $6); }

f_type_expr: f_starttype '(' f_decl_args ')' RET type_expr      { $$ = handleFuncType($3, $6); }

f_starttype: FUNCTION                                           { handleFuncStartType(); }

basic_type: SCALAR  { $$ = new TypeNode(EquelleType(Scalar)); }
          | VECTOR  { $$ = new TypeNode(EquelleType(Vector)); }
          | BOOL    { $$ = new TypeNode(EquelleType(Bool)); }
          | CELL    { $$ = new TypeNode(EquelleType(Cell)); }
          | FACE    { $$ = new TypeNode(EquelleType(Face)); }
          | EDGE    { $$ = new TypeNode(EquelleType(Edge)); }
          | VERTEX  { $$ = new TypeNode(EquelleType(Vertex)); }
          | STRING  { $$ = new TypeNode(EquelleType(String)); }
          ;

f_decl_args: f_decl_args ',' declaration { $$ = $1; $$->addArg($3); }
           | declaration                 { $$ = new FuncArgsDeclNode($1); }
           |                             { $$ = new FuncArgsDeclNode(); }
           ;

quantity: number                   %prec LOWEST { $$ = handleQuantity($1, 0); }
        | number '[' unit_expr ']' %prec UNIT   { $$ = handleQuantity($1, $3); }
        ;

number: INT                     { $$ = handleNumber(numFromString(*($1))); delete $1; }
      | FLOAT                   { $$ = handleNumber(numFromString(*($1))); delete $1; }
      ;

unit_expr: BUILTIN                  { $$ = handleUnit(*($1)); }
         | '(' unit_expr ')'        { $$ = $2; }
         | unit_expr '/' unit_expr  { $$ = handleUnitOp(Divide, $1, $3); }
         | unit_expr '*' unit_expr  { $$ = handleUnitOp(Multiply, $1, $3); }
         | unit_expr '^' INT        { $$ = handleUnitPower($1, numFromString(*($3))); }
         ;

array: '[' f_call_args ']'      { $$ = handleArray($2); }

f_assign_start: ID '(' f_call_args ')' '='    { $$ = handleFuncAssignmentStart(*($1), $3); delete $1; }

f_call_like: BUILTIN '(' f_call_args ')'  { $$ = handleFuncCallLike(*($1), $3); delete $1; }
             | ID '(' f_call_args ')'     { $$ = handleFuncCallLike(*($1), $3); delete $1; }
             ;

f_call_args: f_call_args ',' expr     { $$ = $1; $$->addArg($3); }
           | expr                     { $$ = new FuncArgsNode($1); }
           |                          { $$ = new FuncArgsNode(); }
           ;

loop_start: FOR ID IN ID              { $$ = handleLoopStart(*($2), *($4)); delete $2; delete $4; };


%%

void yyerror(const char* err)
{
    std::cerr << "Parser error near line " << yylineno << ": " << err << std::endl;
}
