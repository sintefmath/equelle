/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef PARSER_HEADER_INCLUDED
#define PARSER_HEADER_INCLUDED

#include <iostream>
#include <string>
#include <sstream>


using namespace std;

void yyerror(const char* s);
extern int yylex();
extern int yylineno;

class Node
{
public:
    Node()
    {}
    virtual ~Node()
    {}
};

typedef Node* NodePtr;


NodePtr createNumber() { return new Node(); }

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

#endif // PARSER_HEADER_INCLUDED
