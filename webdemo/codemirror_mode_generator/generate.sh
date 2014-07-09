#!/bin/bash
# This script runs all the python scripts to generate each part of the Equelle CodeMirror mode
if [[ "$#" -lt 2 ]]; then
    echo "Please provide the equelle_lexer.l and equelle_parser.y file paths as arguments"
    exit 1
fi

# Generate the lexer.js
python generateLexer.py < "$1" > lexer.js
if [[ "$?" -ne 0 ]]; then
    echo "Could not generate lexer.js"
    exit 1
fi

# Generate the parser.js
python generateParser.py < "$2" > parser.js
if [[ "$?" -ne 0 ]]; then
    echo "Could not generate parser.js"
    exit 1
fi

# Generate the style.js
python generateStyle.py > style.js
if [[ "$?" -ne 0 ]]; then
    echo "Could not generate style.js"
    exit 1
fi

# Combine all files into the mode
# The result is sent to stdout
python generateMode.py
