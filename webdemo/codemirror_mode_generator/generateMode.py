#!/usr/bin/python
# ---- This script combines the lexer, parser and styler function into a complete CodeMirror mode ----

import fileinput
import sys

# Open files for reading
lexer = open('lexer.js','r')
parser = open('parser.js','r')
styler = open('style.js','r')

# Read the skeleton from modeSkel.js
skel = open('modeSkel.js','r')
for line in skel:
    il = line.find('##lexer##');
    ip = line.find('##parser##');
    it = line.find('##styler##');
    # Insert the lexer.js file, with indentation, so that it looks nice
    if (il >= 0):
        indent = ' '*il
        for line in lexer:
            sys.stdout.write(indent)
            sys.stdout.write(line)
    # Insert the parser.js file
    elif (ip >= 0):
        indent = ' '*ip
        for line in parser:
            sys.stdout.write(indent)
            sys.stdout.write(line)
    # Insert the styler.js file
    elif (it >= 0):
        indent = ' '*it
        for line in styler:
            sys.stdout.write(indent)
            sys.stdout.write(line)
    # Write all other lines as they are
    else:
        sys.stdout.write(line)
