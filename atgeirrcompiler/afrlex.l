%{
#include <iostream>

#if RETURN_TOKENS
#include "afryacc.hpp"
#define TOKS(x) do { return x; } while(false)
#define TOK(x) do { return x; } while(false)
#else
#define TOKS(x) do { std::cout << x << std::endl; } while(false)
#define TOK(x) do { TOKS(#x); } while(false)
#endif

%}

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
COMMENT  #.*
SINGLES  [:=()+\-*/^<>{},|?\[\]]

%option yylineno
%option outfile="afrlex.cpp"
%option nounput

%%


Collection       { TOK(COLLECTION); }
Of               { TOK(OF); }
On               { TOK(ON); }
Subset           { TOK(SUBSET); }
Scalar           { TOK(SCALAR); }
Vector           { TOK(VECTOR); }
Bool             { TOK(BOOL); }
Cell             { TOK(CELL); }
Face             { TOK(FACE); }
Edge             { TOK(EDGE); }
Vertex           { TOK(VERTEX); }
Function         { TOK(FUNCTION); }
And              { TOK(AND); }
Or               { TOK(OR); }
Not              { TOK(NOT); }
Xor              { TOK(XOR); }
True             { TOK(TRUE); }
False            { TOK(FALSE); }
{BUILTIN}        { yylval.str = new std::string(yytext); TOK(BUILTIN); }
{ID}             { yylval.str = new std::string(yytext); TOK(ID); }
{INT}            { yylval.str = new std::string(yytext); TOK(INT); }
{FLOAT}          { yylval.str = new std::string(yytext); TOK(FLOAT); }
{COMMENT}        { yylval.str = new std::string(yytext); TOK(COMMENT); }
"<="             { TOK(LEQ); }
">="             { TOK(GEQ); }
"=="             { TOK(EQ); }
"!="             { TOK(NEQ); }
"->"             { TOK(RET); }
{SINGLES}        { TOKS(yytext[0]); }
\n               { TOK(EOL); }
{BLANKS}
{INT}{IDCHAR}+   { std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
{FLOAT}{IDCHAR}+ { std::cerr << "Lexer error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
.                {  std::cerr << "Lexer error on line " << yylineno << ": unexpected character \'" << yytext << "\'" << std::endl; }

%%

int yywrap()
{
    return 1;
}
