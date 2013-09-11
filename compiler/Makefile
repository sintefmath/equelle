# Makefile for building the equelle compiler, ec.

# Modify these to suit your setup.
FLEX     = flex
BISON    = bison3
CXX	 = g++-4.7.0
CXXFLAGS = -g -Wall -Wextra -ansi -pedantic


ec:		lex.yy.c y.tab.c
		$(CXX) $(CXXFLAGS) lex.yy.c y.tab.c -o ec

lex.yy.c:	lex.l
		$(FLEX) lex.l

y.tab.c:	yacc.y
		$(BISON) -dy yacc.y


clean:
	rm -f *.o *~ lex.yy.c y.tab.c y.tab.h ec
	rm -rf ec.dSYM
