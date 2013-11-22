/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef COMMON_HEADER_INCLUDED
#define COMMON_HEADER_INCLUDED

#include <string>


// ------ Declarations needed for bison parser ------ 

void yyerror(const char* s);
extern int yylex();
extern int yylineno;

// ------ Utilities used in bison parser ------ 

double numFromString(const std::string& s);
int intFromString(const std::string& s);


#endif // COMMON_HEADER_INCLUDED
