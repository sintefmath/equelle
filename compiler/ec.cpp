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
#include "CommandLineOptions.hpp"

#include <iostream>

extern int yylex();
extern int yyparse();
extern FILE * yyin;

/**
 * Trivial class to properly close yyin
 */
class YYInOwner {
public:
	YYInOwner(const std::string filename_) {
		yyin = fopen(filename_.c_str(),"r");
	}
	~YYInOwner() {
	    fclose(yyin);
	    yyin = NULL;
	}
};


int main(int argc, char** argv)
{
	CommandLineOptions options;
	boost::program_options::variables_map cli_vars;
	boost::shared_ptr<YYInOwner> yyin_owner;

	//Parse commandline
	try {
		cli_vars = options.parse(argc, argv);

		if (cli_vars.count("help")) {
			options.printOptions();
			return -1;
		}
	}
	catch (const std::exception& e) {
		std::cout << "Error parsing options: ";
		std::cout << e.what() << std::endl;
		return -1;
	}

	//Get input file
	if (cli_vars.count("input")) {
		std::string infile = cli_vars["input"].as<std::string>();
		if (infile != "-") { //"-" signifies use stdin
			yyin_owner.reset(new YYInOwner(infile));
		}
	}

	//Parse equelle program
    yyparse();

    //Write output
    std::string backend = cli_vars["backend"].as<std::string>();
    if (backend == "ast") {
        PrintASTVisitor v;
        SymbolTable::program()->accept(v);
    }
    else if (backend == "equelle_ast") {
        PrintEquelleASTVisitor v;
        SymbolTable::program()->accept(v);
    }
    else if (backend == "cpu") {
        PrintCPUBackendASTVisitor v;
        SymbolTable::program()->accept(v);
    }
    else if (backend == "cuda") {
        PrintCUDABackendASTVisitor v;
        SymbolTable::program()->accept(v);
    }
    else if (backend == "mrst") {
        PrintMRSTBackendASTVisitor v;
        SymbolTable::program()->accept(v);
    } else if(backend == "MPI") {
        PrintMPIBackendASTVisitor v;
        SymbolTable::program()->accept(v);
    }
    else {
        std::cerr << "Unknown back-end choice: " << backend << '\n';
    }

    return 0;
}
