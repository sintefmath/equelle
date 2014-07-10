CodeMirror Equelle language mode generator scripts
==================================================

The script in this folder is used to create a CodeMirror mode for syntax highlighting of the Equelle language in the CodeMirror Editor.

To generate the code, run the "generate.py" script,t he result is printed to stdout. E.g.:

    ./generate.py > ../srv/files/js/equelleMode.js


Token styling
-------------
To modify the styling of the different tokens, see the styles.txt file. Each line of tokens will be given the css-class defined by the preceeding {{style}} directive, or "keyword" as default.
