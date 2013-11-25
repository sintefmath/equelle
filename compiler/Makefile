# Makefile for building the equelle compiler, ec.

# Modify these to suit your setup.
FLEX     = flex
BISON    = bison
CXX	 = g++
# The -Wno-sign-compare is because flex generates a scanner with signed/unsigned comparison!
CXXFLAGS = -std=c++0x -g -Wall -Wextra -Wno-sign-compare -pedantic

ECHEADERS = Common.hpp EquelleType.hpp SymbolTable.hpp ASTNodes.hpp ParseActions.hpp ASTVisitorInterface.hpp PrintASTVisitor.hpp PrintEquelleASTVisitor.hpp PrintCPUBackendASTVisitor.hpp PrintMRSTBackendASTVisitor.hpp NodeInterface.hpp
ECOBJS = equelle_lexer_tok.o equelle_parser.o Common.o EquelleType.o SymbolTable.o ParseActions.o PrintASTVisitor.o PrintEquelleASTVisitor.o PrintCPUBackendASTVisitor.o PrintMRSTBackendASTVisitor.o

all:		ec el

ec:		$(ECHEADERS) $(ECOBJS)
		$(CXX) $(CXXFLAGS) $(ECOBJS) ec.cpp -o ec

el:		equelle_lexer.o el.cpp
		$(CXX) $(CXXFLAGS) equelle_lexer.o el.cpp -o el

# This is needed because ec and el need different settings for the RETURN_TOKENS macro.
equelle_lexer_tok.o: equelle_lexer.cpp equelle_parser.cpp
		$(CXX) $(CXXFLAGS) -DRETURN_TOKENS=1 equelle_lexer.cpp -c -o equelle_lexer_tok.o

equelle_lexer.cpp:	equelle_lexer.l
		$(FLEX) equelle_lexer.l

equelle_parser.cpp:	equelle_parser.y ParseActions.hpp
		$(BISON) -v equelle_parser.y


clean:
		rm -f *.o *~ equelle_lexer.cpp equelle_parser.cpp equelle_parser.hpp equelle_parser.output ec el
		rm -rf ec.dSYM el.dSYM
