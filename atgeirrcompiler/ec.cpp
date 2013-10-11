/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

extern int yylex();
extern int yyparse();

#include "SymbolTable.hpp"
#include "PrintASTVisitor.hpp"
#include "ASTNodes.hpp"

int main()
{
    yyparse();
    PrintASTVisitor v;
    SymbolTable::program()->accept(v);
}
