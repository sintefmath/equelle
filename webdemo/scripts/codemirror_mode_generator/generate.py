#!/usr/bin/python
# ---- This script generates a complete CodeMirror mode for the Equelle language, based on the FLEX configuration file ----
import re
import fileinput
import sys

## ---------- LEXER GENERATION START ---------- ##
# Replaces recursively defined regular expressions, and newlines with end-of-line
def regexReplace(string, replaces):
    matches = re.findall(r'\{(\w+)\}',string)
    for m in matches:
        string = string.replace('{'+m+'}',replaces[m])
    string = string.replace('\\n','$')
    return string

# Parse regular expression definitions
regexes = {}
def handleDefinition(line):
    split = line.partition(' ')
    name = split[0]
    regex = split[2].lstrip()
    # Insert substituted regex into list
    regexes[name] = regexReplace(regex,regexes)

# Parse the token rules
rules = []
def handleRule(line):
    split = line.partition(' ')
    regex = split[0].strip('"')
    action = split[2].lstrip()
    # Subsititute previously defined regexes
    substituted = regexReplace(regex,regexes)
    # Add this rule to the list
    rules.append((substituted, action))

# Read the equelle_lexer.l and parse
flexfile = open('/equelle/src/compiler/equelle_lexer.l')
section = 'definitions'
comment = False
for line in flexfile:
    # Handle comment sections
    if (not comment and line.startswith('%{')):
        comment = True
    elif (comment and line.startswith('%}')):
        comment = False
    # Handle sections of the FLEX file
    elif (not comment and line.startswith('%%')):
        if (section == 'definitions'):
            section = 'rules'
        elif (section == 'rules'):
            section = 'user code'
    # Handle the lines that are not comments and not FLEX statements
    elif (not comment and not line.startswith('%')):
        line = line.replace('\n','').strip()
        if (len(line)):
            # Parse the non-empty lines
            if (section == 'definitions'):
                handleDefinition(line)
            elif (section == 'rules'):
                handleRule(line)

# Convert a non-empty FLEX action into a JavaScript action
def convertAction(action):
    # This is inside the function body, the variables yytext and yylineno are available
    # And the function is going to return the object r to the parent lexer
    # The STORE macro
    if (action == 'STORE'):
        return 'r.text = yytext; '
    # The TOK macro
    m = re.match(r'TOK\((.*)\)',action)
    if (m):
        return 'r.name = \''+m.group(1)+'\'; '
    # The TOKS macro
    m = re.match(r'TOKS\((.*)\)',action)
    if (m):
        return 'r.name = '+m.group(1)+'; '
    # Error reporting
    if (action.startswith('std::cerr')):
        parts = action.split('<<')
        trimmed = [ str.strip(p) for p in parts ]
        if (trimmed[0] == 'std::cerr' and trimmed[-1] == 'std::endl'):
            # We can convert this into a throw
            r = 'throw ('
            delimiter = ''
            for part in trimmed[1:-1]:
                # A string
                if (part[0] == '"' and part[-1] == '"'):
                    r += delimiter+part
                # A variable
                elif (part == 'yylineno'):
                    r += delimiter+part
                elif (part == 'yytext'):
                    r += delimiter+part
                # Something else
                else:
                    continue
                delimiter = ' + '
            r += '); return undefined; '
            return r
        else:
            # This is an unexpected format
            return '';
    # Not translated actions
    return ''

# Create JavaScript function from FLEX action
def createActionFunction(action):
    # Create the return token object
    ret = 'var r = {}; '
    # Always store the matched text, so we can use it elsewhere
    ret += 'r.match = yytext; ';
    # Parse actions
    for a in action.split(';'):
        a = a.strip()
        if (len(a)):
            ret += convertAction(a)
    # Return the token object
    ret += 'return r;'
    return ret
## ---------- LEXER GENERATION END   ---------- ##



## ---------- STYLE GENERATION START ---------- ##
stylefile = open('styles.txt')
# Read the styles.txt and parse
cases = { 'keyword': [] }
style = 'keyword'
styleToken = r'{{(.*)}}'
for line in stylefile:
    line = line.replace('\n','').strip()
    if (len(line)):
        m = re.search(styleToken, line);
        if (m):
            # Set current style when passing a {{token}}
            style = m.group(1)
            if (not style in cases):
                cases[style] = []
        else:
            # Add this token to current style
            cases[style].append(line)
## ---------- STYLE GENERATION END   ---------- ##

## ---------- COMPLETE MODE GENERATION START --------##
# Read the skeleton from mode.js
skel = open('mode.js','r')
for line in skel:
    ilr = line.find('##lexer_regexes##');
    ila = line.find('##lexer_actions##');
    isc = line.find('##style_cases##');
    # Insert the lexer regexes into the skeleton, with indentation, so that it looks nice
    if (ilr >= 0):
        separator = ' '
        indent = ' '*ilr
        for rule in rules:
            sys.stdout.write(indent)
            sys.stdout.write(separator)
            sys.stdout.write('/^')
            sys.stdout.write(rule[0])
            sys.stdout.write('/\n')
            separator = ','
    # Insert the actions into the skeleton
    elif (ila >= 0):
        separator = ' '
        indent = ' '*ila
        for rule in rules:
            action = rule[1].lstrip('{').rstrip('}').strip()
            funcBody = createActionFunction(action)
            sys.stdout.write(indent)
            sys.stdout.write(separator)
            sys.stdout.write('function(yytext,yylineno) { ')
            sys.stdout.write(funcBody)
            sys.stdout.write(' }\n')
            separator = ','
    # Insert cases, with indentation, so it looks nice
    elif (isc >= 0):
        indent = ' '*isc
        for style, tokens in cases.iteritems():
            # Output all tokens that correspond to same style
            for token in tokens:
                sys.stdout.write(indent)
                sys.stdout.write('case "')
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
