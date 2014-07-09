#!/usr/bin/python
# ---- This script generates the styler function that uses a switch-case statement to style incoming Equelle tokens with a css class ----

# Helper function for constructing the cases
cases = {}
def addStyle(tokens, style):
    if (style in cases):
        cases[style].extend(tokens)
    else:
        cases[style] = tokens

##-- Define all styling classes here --##

addStyle(['STRING_LITERAL'],'string')

addStyle(['COLLECTION','SEQUENCE','ARRAY'],'type');
addStyle(['SCALAR','VECTOR','BOOL','CELL','FACE','EDGE','VERTEX'],'type');

addStyle(['OF','ON','EXTEND','SUBSET','MUTABLE'],'keyword');

addStyle(['BUILTIN','FUNCTION'],'builtin');

addStyle(['AND','OR','NOT','XOR'],'keyword');

addStyle(['TRUE','FALSE'],'atom');

addStyle(['FOR','IN'],'keyword');

addStyle(['ID'],'variable');

addStyle(['INT','FLOAT'],'number');

addStyle(['COMMENT'],'comment');

addStyle(['LEQ','GEQ','EQ','NEQ',':','=','+','-','/','^','<','>','?'],'operator');

addStyle(['(',')','[',']','{','}','|'],'bracket');

addStyle(['$','@'],'keyword');

##-- End of styling classes definitions --##

import fileinput
import sys
# Read the skeleton from styleSkel.js
skel = open('styleSkel.js','r')
for line in skel:
    i = line.find('##cases##');
    # Insert cases, with indentation, so it looks nice
    if (i >= 0):
        indent = ' '*i
        for style, tokens in cases.iteritems():
            # Output all tokens that correspond to same style
            for token in tokens:
                sys.stdout.write(indent)
                sys.stdout.write('case: "')
                sys.stdout.write(token)
                sys.stdout.write('":\n')
            # Return the style of these tokens
            sys.stdout.write(indent)
            sys.stdout.write('return "')
            sys.stdout.write(style)
            sys.stdout.write('";\n\n')
    # Write all other lines as they are
    else:
        sys.stdout.write(line)
