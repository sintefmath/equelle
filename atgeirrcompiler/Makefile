# Makefile for building the equelle compiler, ec.

# Modify these to suit your setup.
FLEX     = flex
BISON    = bison3
CXX	 = g++-4.7.0
# The -Wno-sign-compare is because flex generates a scanner with signed/unsigned comparison!
CXXFLAGS = -g -Wall -Wextra -Wno-sign-compare -ansi -pedantic

all:		ac al

ac:		afrlex.cpp afryacc.c
		$(CXX) $(CXXFLAGS) afrlex.cpp afryacc.c -o ac

afrlex.cpp:	afrlex.l
		$(FLEX) afrlex.l

afryacc.c:	afryacc.y
		$(BISON) -dy afryacc.y

al:		afrlex.cpp
		$(CXX) $(CXXFLAGS) afrlex.cpp al.cpp -o al

clean:
	rm -f *.o *~ afrlex.cpp afryacc.c afryacc.h ac al
	rm -rf ac.dSYM al.dSYM
