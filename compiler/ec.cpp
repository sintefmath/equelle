/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "SymbolTable.hpp"
#include "CheckASTVisitor.hpp"
#include "PrintASTVisitor.hpp"
#include "PrintEquelleASTVisitor.hpp"
#include "PrintCPUBackendASTVisitor.hpp"
#include "PrintMRSTBackendASTVisitor.hpp"
#include "PrintCUDABackendASTVisitor.hpp"
#include "PrintMPIBackendASTVisitor.hpp"
#include "PrintIOVisitor.hpp"
#include "ASTNodes.hpp"
#include "CommandLineOptions.hpp"
#include "ASTRewriter.hpp"

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
                if (!yyin) {
                    throw std::runtime_error("Input file not found.");
                }
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
			std::cout << "Usage: ./eq <options>" << std::endl;
			std::cout << "The following options are supported:" << std::endl;
			options.printOptions();
			return -1;
		}
		if (cli_vars.count("verbose")) {
			options.printVars(cli_vars);
		}
	}
	catch (const std::exception& e) {
		std::cerr << "Usage: ./eq <options>" << std::endl;
		std::cerr << "The following options are supported:" << std::endl;
		options.printOptions();
		std::cerr << std::endl;
        std::cerr << "Error parsing options: ";
        std::cerr << e.what() << std::endl;
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

    // Check AST (and build symbol table)
    const bool ignore_dimensions = cli_vars.count("nondimensional");
    CheckASTVisitor check(ignore_dimensions);
    SymbolTable::program()->accept(check);

    // Dump the SymbolTable
    if (cli_vars["dump"].as<std::string>() == "symboltable") {
        SymbolTable::dump();
    }
    //Write output
    else {
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
            // Check if we use the Cartesian dialect
            const bool use_cartesian = cli_vars.count("cartesian");
            PrintCPUBackendASTVisitor v(use_cartesian);
            SymbolTable::program()->accept(v);
        }
        else if (backend == "cuda") {
            PrintCUDABackendASTVisitor v;
            SymbolTable::program()->accept(v);
        }
        else if (backend == "cuda-dev") {
            ASTRewriter rewriter;
            rewriter.rewrite(SymbolTable::program(),0);
            PrintCUDABackendASTVisitor v;
            SymbolTable::program()->accept(v);
        }
        else if (backend == "mrst") {
            PrintMRSTBackendASTVisitor v;
            SymbolTable::program()->accept(v);
        }
        else if(backend == "MPI") {
            PrintMPIBackendASTVisitor v;
            SymbolTable::program()->accept(v);
        }
        else if(backend == "io") {
            PrintIOVisitor v;
            SymbolTable::program()->accept(v);
        }
        else {
            std::cerr << "Unknown back-end choice: " << backend << '\n';
        }
    }

    //This assumes that the printing went well.
    if (check.isValid())
    {
    	return 0;
    }
    else
    {
    	return -1;
    }
}
