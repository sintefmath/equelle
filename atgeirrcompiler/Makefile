# Makefile for building the equelle compiler, ec.

# Modify these to suit your setup.
FLEX     = flex
BISON    = bison3
CXX	 = g++-4.7.0
# The -Wno-sign-compare is because flex generates a scanner with signed/unsigned comparison!
CXXFLAGS = -std=c++11 -g -Wall -Wextra -Wno-sign-compare -pedantic

all:		ac al

ac:		afrlex.cpp afryacc.cpp
		$(CXX) $(CXXFLAGS) -DRETURN_TOKENS=1 afrlex.cpp afryacc.cpp ac.cpp -o ac

afrlex.cpp:	afrlex.l
		$(FLEX) afrlex.l

afryacc.cpp:	afryacc.y Parser.hpp
		$(BISON) -v afryacc.y

al:		afrlex.cpp
		$(CXX) $(CXXFLAGS) afrlex.cpp al.cpp -o al

clean:
	rm -f *.o *~ afrlex.cpp afryacc.cpp afryacc.hpp afryacc.output ac al
	rm -rf ac.dSYM al.dSYM
