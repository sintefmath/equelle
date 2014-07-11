CodeMirror Equelle language support
===================================
The scripts in this folder is used to create a CodeMirror mode for syntax highlighting of, and code-completion for the Equelle language in the CodeMirror Editor.

Mode generation script
----------------------

To generate the code, run the "generate.py" script, the result is printed to stdout. E.g.:

    ./generate.py > ../srv/files/js/equelleMode.js



Code completion script
----------------------
To generate the code, run the "generateHints.py" script the same way as the mode generation script. E.g.:

    ./generateHints.py > ../srv/files/js/equelleHints.js


Token styling
-------------
To modify the styling of the different tokens, see the styles.txt file. Each line of tokens will be given the css-class defined by the preceeding {{style}} directive, or "keyword" as default.
