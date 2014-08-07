%{
/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include <iostream>

#if RETURN_TOKENS
#include "equelle_parser.hpp"
#define TOKS(x) do { return x; } while(false)
#define TOK(x) do { return x; } while(false)
#define STORE do { yylval.str = new std::string(yytext); } while(false)
#else
#define TOKS(x) do { std::cout << x << std::endl; } while(false)
#define TOK(x) do { TOKS(#x); } while(false)
#define STORE do { std::cout << "\'" << yytext << "\'   "; } while(false)
#endif

%}

COMMENT  #.*
DIGIT    [0-9]
LCASE    [a-z]
UCASE    [A-Z]
LETTER   [a-zA-Z]
USCORE   [_]
IDCHAR   [0-9a-zA-Z_]
ID       {LCASE}{IDCHAR}*
BUILTIN  {UCASE}{IDCHAR}*
INT      {DIGIT}+
FLOAT    {INT}[.]{INT}
BLANKS   [\t ]+
STRING   \"(\\.|[^\\"])*\"
SINGLES  [$@:=()+\-*/^<>{},|?\[\]]
ELLIPSIS [.][.][.]
LINECONT {ELLIPSIS}[\t ]*\n

%option yylineno
%option nounput

%%

{STRING}          { STORE; TOK(STRING_LITERAL); }
Collection        { TOK(COLLECTION); }
Stencil           { TOK(STENCIL); }
Sequence          { TOK(SEQUENCE); }
Array             { TOK(ARRAY); }
Of                { TOK(OF); }
On                { TOK(ON); }
Extend            { TOK(EXTEND); }
Subset            { TOK(SUBSET); }
Scalar            { TOK(SCALAR); }
Vector            { TOK(VECTOR); }
Bool              { TOK(BOOL); }
Cell              { TOK(CELL); }
Face              { TOK(FACE); }
Edge              { TOK(EDGE); }
Vertex            { TOK(VERTEX); }
Function          { TOK(FUNCTION); }
And               { TOK(AND); }
Or                { TOK(OR); }
Not               { TOK(NOT); }
Xor               { TOK(XOR); }
True              { TOK(TRUE); }
False             { TOK(FALSE); }
For               { TOK(FOR); }
In                { TOK(IN); }
Mutable           { TOK(MUTABLE); }
{BUILTIN}         { STORE; TOK(BUILTIN); }
{ID}              { STORE; TOK(ID); }
{INT}             { STORE; TOK(INT); }
{FLOAT}           { STORE; TOK(FLOAT); }
{COMMENT}         { STORE; TOK(COMMENT); }
"<="              { TOK(LEQ); }
">="              { TOK(GEQ); }
"=="              { TOK(EQ); }
"!="              { TOK(NEQ); }
"->"              { TOK(RET); }
{SINGLES}         { TOKS(yytext[0]); }
\n                { TOK(EOL); }
{LINECONT}
{BLANKS}
{INT}{IDCHAR}+   { std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
{FLOAT}{IDCHAR}+ { std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
.                { std::cerr << "Lexer error on line " << yylineno << ": unexpected character \'" << yytext << "\'" << std::endl; }

%%

int yywrap()
{
    return 1;
}
