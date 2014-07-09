#!/usr/bin/python
# ---- This script generates the parser function that handles all parsing of the Equelle tokens in the mode ----

import fileinput
import sys
# Read the skeleton from styleSkel.js
skel = open('parserSkel.js','r')
for line in skel:
    #i = line.find('##cases##');
    # Insert cases, with indentation, so it looks nice
    #if (i >= 0):
    #    indent = ' '*i
    #    pass
    # Write all other lines as they are
    #else:
        sys.stdout.write(line)
