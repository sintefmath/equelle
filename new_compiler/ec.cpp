/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

extern int yylex();
extern int yyparse();

#include "SymbolTable.hpp"
#include "PrintASTVisitor.hpp"
#include "PrintEquelleASTVisitor.hpp"
#include "PrintCPUBackendASTVisitor.hpp"
#include "ASTNodes.hpp"

int main(int argc, char** argv)
{
    yyparse();
    PrintASTVisitor v0;
    PrintEquelleASTVisitor v1;
    PrintCPUBackendASTVisitor v2;
    int which = 2;
    if (argc > 1) {
        which = std::atoi(argv[1]);
    }
    switch (which) {
    case 0:
        SymbolTable::program()->accept(v0);
        break;
    case 1:
        SymbolTable::program()->accept(v1);
        break;
    case 2:
        SymbolTable::program()->accept(v2);
        break;
    }
}