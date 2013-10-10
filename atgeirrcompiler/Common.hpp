/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef COMMON_HEADER_INCLUDED
#define COMMON_HEADER_INCLUDED

#include <string>
#include <sstream>


// ------ Declarations needed for bison parser ------ 

void yyerror(const char* s);
extern int yylex();
extern int yylineno;

// ------ Utilities used in bison parser ------ 

inline double numFromString(const std::string& s)
{
    std::istringstream iss(s);
    double num;
    iss >> num;
    if (!iss) {
        yyerror("error in string-to-number conversion.");
    }
    return num;
}


#endif // COMMON_HEADER_INCLUDED
