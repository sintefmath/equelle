%{
#include <iostream>
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
LE       <=
GE       >=
EQ       ==
RETURNS  ->
SINGLES  [:=()+\-*/^<>{},|]

%option yylineno
%option outfile="afrlex.cpp"
%option nounput

%%


Collection       { std::cout << "COLLECTION" << std::endl; }
Of               { std::cout << "OF" << std::endl; }
On               { std::cout << "ON" << std::endl; }
Subset           { std::cout << "SUBSET" << std::endl; }
Scalar           { std::cout << "SCALAR" << std::endl; }
Vector           { std::cout << "VECTOR" << std::endl; }
Bool             { std::cout << "BOOL" << std::endl; }
Function         { std::cout << "FUNCTION" << std::endl; }
Cell             { std::cout << "CELL" << std::endl; }
Face             { std::cout << "FACE" << std::endl; }
Edge             { std::cout << "EDGE" << std::endl; }
Vertex           { std::cout << "VERTEX" << std::endl; }
Return           { std::cout << "RETURN" << std::endl; }
{BUILTIN}        { std::cout << "BUILTIN" << std::endl; }
{ID}             { std::cout << "ID" << std::endl; }
{INT}            { std::cout << "INT" << std::endl; }
{FLOAT}          { std::cout << "FLOAT" << std::endl; }
{COMMENT}        { std::cout << "COMMENT" << std::endl; }
{LE}             { std::cout << "LE" << std::endl; }
{GE}             { std::cout << "GE" << std::endl; }
{EQ}             { std::cout << "EQ" << std::endl; }
{RETURNS}        { std::cout << "RETURNS" << std::endl; }
{SINGLES}        { std::cout << yytext[0] << std::endl; }
\n               { std::cout << "EOL" << std::endl; }
{BLANKS}
{INT}{IDCHAR}+   { std::cout << "Error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
{FLOAT}{IDCHAR}+ { std::cout << "Error on line " << yylineno << ": this is not a number \'" << yytext << "\'" << std::endl; }
.                {  std::cout << "Error on line " << yylineno << ": unexpected character \'" << yytext << "\'" << std::endl; }

%%

int yywrap()
{
    return 1;
}
