CodeMirror Equelle language mode generator scripts
==================================================

The scripts in this folder are used to create a CodeMirror mode for syntax highlighting of the Equelle language in the CodeMirror Editor.

To generate the code, run the generate.sh script passing it the paths to the "equelle\_lexer.l" and "equelle\_parser.y" files as arguments.
The result is printed to stdout. E.g.:

    ./generate.sh ../../compiler/equelle_lexer.l ../../compiler/equelle_parser.y > ../srv/files/js/equelleMode.js


Token styling
-------------
To modify the styling of the different tokens, see the generateStyle.py file.
