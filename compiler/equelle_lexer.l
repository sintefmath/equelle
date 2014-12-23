%{
/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include <iostream>
#include "equelle_parser.hpp"

#if RETURN_TOKENS
#define TOKS(x) do { return x; } while(false)
#define TOK(x) do { return x; } while(false)
#define STORE do { yylval.str = new std::string(yytext); } while(false)
#else
#define TOKS(x) do { std::cout << x << std::endl; } while(false)
#define TOK(x) do { TOKS(#x); } while(false)
#define STORE do { std::cout << "\'" << yytext << "\'   "; } while(false)
#endif

bool increment_yylineno = false;

#define INCIF do { if (increment_yylineno) ++yylineno; increment_yylineno = false; } while(false)
#define NEWL do { --yylineno; increment_yylineno = true; } while(false)

%}

COMMENT  #.*
FLOAT    (0|[1-9][0-9]*)?\.?[0-9]+(e[+-]?(0|[1-9][0-9]*))?
DIGIT    [0-9]
LCASE    [a-z]
UCASE    [A-Z]
LETTER   [a-zA-Z]
USCORE   [_]
IDCHAR   [0-9a-zA-Z_]
ID       {LCASE}{IDCHAR}*
BUILTIN  {UCASE}{IDCHAR}*
INT      {DIGIT}+
BLANKS   [\t ]+
STRINGL  \"(\\.|[^\\"])*\"
SINGLES  [$@:=()+\-*/^<>{},|?\[\]]
ELLIPSIS [.][.][.]
LINECONT {ELLIPSIS}[\t ]*\n

%option yylineno
%option nounput

%%

{STRINGL}         { INCIF; STORE; TOK(STRING_LITERAL); }
Collection        { INCIF; TOK(COLLECTION); }
Stencil           { INCIF; TOK(STENCIL); }
Sequence          { INCIF; TOK(SEQUENCE); }
Array             { INCIF; TOK(ARRAY); }
Of                { INCIF; TOK(OF); }
On                { INCIF; TOK(ON); }
Extend            { INCIF; TOK(EXTEND); }
Subset            { INCIF; TOK(SUBSET); }
Scalar            { INCIF; TOK(SCALAR); }
Vector            { INCIF; TOK(VECTOR); }
Bool              { INCIF; TOK(BOOL); }
Cell              { INCIF; TOK(CELL); }
Face              { INCIF; TOK(FACE); }
Edge              { INCIF; TOK(EDGE); }
Vertex            { INCIF; TOK(VERTEX); }
String            { INCIF; TOK(STRING); }
Function          { INCIF; TOK(FUNCTION); }
And               { INCIF; TOK(AND); }
Or                { INCIF; TOK(OR); }
Not               { INCIF; TOK(NOT); }
Xor               { INCIF; TOK(XOR); }
True              { INCIF; TOK(TRUE); }
False             { INCIF; TOK(FALSE); }
For               { INCIF; TOK(FOR); }
In                { INCIF; TOK(IN); }
Mutable           { INCIF; TOK(MUTABLE); }
{BUILTIN}         { INCIF; STORE; TOK(BUILTIN); }
{ID}              { INCIF; STORE; TOK(ID); }
{INT}             { INCIF; STORE; TOK(INT); }
{FLOAT}           { INCIF; STORE; TOK(FLOAT); }
{COMMENT}         { INCIF; STORE; TOK(COMMENT); }
"<="              { INCIF; TOK(LEQ); }
">="              { INCIF; TOK(GEQ); }
"=="              { INCIF; TOK(EQ); }
"!="              { INCIF; TOK(NEQ); }
"->"              { INCIF; TOK(RET); }
{SINGLES}         { INCIF; TOKS(yytext[0]); }
\n                { INCIF; NEWL; TOK(EOL); }
{LINECONT}        { INCIF; }
{BLANKS}          { INCIF; }
{INT}{IDCHAR}+    { INCIF; std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
{FLOAT}{IDCHAR}+  { INCIF; std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
.                 { INCIF; std::cerr << "Lexer error on line " << yylineno << ": unexpected character \'" << yytext << "\'" << std::endl; }

%%

int yywrap()
{
    return 1;
}
