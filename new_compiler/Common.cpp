/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "Common.hpp"

#include <string>
#include <sstream>


// ------ Utilities used in bison parser ------ 

double numFromString(const std::string& s)
{
    std::istringstream iss(s);
    double num;
    iss >> num;
    if (!iss) {
        yyerror("internal compiler error in string-to-number conversion.");
    }
    return num;
}

