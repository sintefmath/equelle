/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

extern int yylex();
extern int yyparse();

#include "SymbolTable.hpp"
#include "PrintASTVisitor.hpp"
#include "PrintEquelleASTVisitor.hpp"
#include "PrintCPUBackendASTVisitor.hpp"
#include "PrintMRSTBackendASTVisitor.hpp"
#include "PrintCUDABackendASTVisitor.hpp"
#include "PrintMPIBackendASTVisitor.hpp"
#include "ASTNodes.hpp"
#include <iostream>

int main(int argc, char** argv)
{    
    yyparse();
    PrintASTVisitor v0;
    PrintEquelleASTVisitor v1;
    PrintCPUBackendASTVisitor v2;
    PrintMRSTBackendASTVisitor v3;
    PrintCUDABackendASTVisitor v4;
    PrintMPIBackendASTVisitor v5;
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
    case 3:
        SymbolTable::program()->accept(v3);
        break;
    case 4:
    	SymbolTable::program()->accept(v4);
    	break;
    case 5:
        SymbolTable::program()->accept(v5);
        break;
    default:
        std::cerr << "Unknown back-end choice: " << which << '\n';
    }
}
