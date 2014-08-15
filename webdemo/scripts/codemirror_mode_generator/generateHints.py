#!/usr/bin/python
# ---- This script generates code-completion hints, based on the Bison configuration file ----
import re
import fileinput
import sys
import shlex

# Bison grammar rules
nodes = {}
currentNode = None
nodeStartRegx = r'([^:]*):\s*((\'{|[^%{])*)'
nodeOrRegx = r'\|((\'{|[^{])*)'
def handleRule(line):
    global currentNode
    # Make a list off all nodes, and their expected children
    if (line.startswith(';')):
        currentNode = None
    elif (line.startswith('|')):
        m = re.match(nodeOrRegx, line)
        expects = m.group(1).strip()
        nodes[currentNode].append(expects)
    else:
        m = re.match(nodeStartRegx, line)
        name = m.group(1).strip()
        expects = m.group(2).strip()
        nodes[name] = [ expects ]
        currentNode = name


# Read the equelle_parser.y and parse
bisonfile = open('/equelle/src/compiler/equelle_parser.y')
section = 'bison declarations'
comment = False
for line in bisonfile:
    # Handle comment sections
    if (not comment and line.startswith('%{')):
        comment = True
    elif (comment and line.startswith('%}')):
        comment = False
    # Handle sections of the FLEX file
    elif (not comment and line.startswith('%%')):
        if (section == 'bison declarations'):
            section = 'rules'
        elif (section == 'rules'):
            section = 'user code'
    # Handle the lines that are not comments and not FLEX statements
    elif (not comment):
        line = line.replace('\n','').strip()
        if (len(line)):
            # Parse the non-empty lines
            if (section == 'rules'):
                handleRule(line)


# We also need a list of what to convert the tokens into when we put them into the editor, we get this from the "equelle_lexer.l" file
# We can only insert real text, so only read lines that are not defined by a RegExp
tokenReplacements = {}
def handleFlex(line):
    if (not line.startswith('{')):
        split = line.partition(' ')
        replacement = split[0].strip('"')
        action = split[2].lstrip().lstrip('{').rstrip('}').strip()
        # Find lines that define tokens by the TOK macro
        m = re.match(r'TOK\((.*)\)',action)
        if (m):
            tokenReplacements[m.group(1)] = replacement

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
            if (section == 'rules'):
                handleFlex(line)


# The rules and tokens have been parsed, we are now ready to build the list of statements that the hinter can use
# Since the Equelle web editor doesn't really understand lines and blocks, we have to cheat a little bit with the grammar
start = 'statement' # This is our root-node
nodes['block'] = [ "'{' EOL EOL '}'" ] # This is what we will insert when a block is requested
# ... and expressions can be so general that we need to suggest just a few of them*/
#nodes['expr'] = [ "ID", "number", "STRING_LITERAL", "function_call" ]

supportedNodes = {}
def addNode(name):
    supportedNodes[name] = None
    expects = []
    # Read all available children
    for expected in nodes[name]:
        # Parse the expected parts
        parts = []
        for part in shlex.split(expected):
            if part in nodes:
                # A new node
                parts.append({ 'type': 'node', 'name': part })
                if (part != name and part not in supportedNodes):
                    addNode(part)
            else:
                # Thus must be an Equelle token
                parts.append({ 'type': 'token', 'name': part })
        expects.append(parts)
    supportedNodes[name] = expects

#Start with our root-node
addNode(start)

# Write out node-function
def writeStatementTokens(tokens, indent):
    if (len(tokens)>0):
        toki = range(len(tokens))
        tokns = ','.join(['t'+str(i+1) for i in toki])
        # Predeclare tokens
        sys.stdout.write(indent)
        sys.stdout.write('var '+tokns+';\n')
        # Write actual tokens
        for i in toki:
            # Find next return
            next = 'end'
            if (i+1 < len(tokens)):
                next = 't'+str(i+2)
            # Write token
            sys.stdout.write(indent)
            sys.stdout.write('t'+str(i+1)+' = ')
            if (tokens[i]['type'] == 'node'):
                sys.stdout.write('_.partial(nodefun_'+tokens[i]['name']+', function() { return '+next+'() })')
            else: #type == 'token'
                sys.stdout.write('function() { return {\''+tokens[i]['name']+'\': '+next+'} }')
            sys.stdout.write(';\n')
        # Return first token
        sys.stdout.write(indent)
        sys.stdout.write('return t1();\n')
    else:
        # This token just returns parent
        sys.stdout.write(indent)
        sys.stdout.write('return end();\n')

def writeNodeFunction(node, indent):
    subindent = indent+(' '*4)
    subsubindent = subindent+(' '*4)
    # Separat simple statements from repeated ones (where a node calls itself recursively as first token)
    simple = {}
    repeat = {}
    for statement in supportedNodes[node]:
        if (len(statement) > 0 and statement[0]['type'] == 'node' and statement[0]['name'] == node):
            statement.pop(0)
            repeat['r'+str(len(repeat)+1)] = statement
        else:
            simple['s'+str(len(simple)+1)] = statement
    #-- Write start of node-function --
    sys.stdout.write(indent)
    sys.stdout.write('nodefun_'+node)
    sys.stdout.write(' = function(p) {\n')
    #- Predeclare the end return -
    sys.stdout.write(subindent)
    sys.stdout.write('var end;\n')
    #- Write optional repeated statements first -
    if (len(repeat) > 0):
        repeats = ','.join(repeat.keys())
        # Predeclare variables
        sys.stdout.write(subindent)
        sys.stdout.write('var '+repeats+';\n')
        # Write the statements
        for n, toks in repeat.iteritems():
            sys.stdout.write(subindent)
            sys.stdout.write('var '+n+' = function() {\n')
            writeStatementTokens(toks, subsubindent)
            sys.stdout.write(subindent)
            sys.stdout.write('};\n')
            sys.stdout.write(subindent)
            sys.stdout.write('end = _.partial(combine, ['+repeats+',p]);\n')
    else:
        # The only end statemt is to go back to parent
        sys.stdout.write(subindent)
        sys.stdout.write('end = p;\n')
    #- Write normal simple statments -
    simples = ','.join(simple.keys())
    # Predeclare variables
    sys.stdout.write(subindent)
    sys.stdout.write('var '+simples+';\n')
    for n, toks in simple.iteritems():
        sys.stdout.write(subindent)
        sys.stdout.write('var '+n+' = function() {\n')
        writeStatementTokens(toks, subsubindent)
        sys.stdout.write(subindent)
        sys.stdout.write('};\n')
    #- Return combination of all statements, or just the one -
    sys.stdout.write(subindent)
    sys.stdout.write('return ')
    if (len(simple) > 1):
        sys.stdout.write('combine(['+simples+'])')
    else:
        sys.stdout.write(simples+'()')
    sys.stdout.write(';\n')
    #-- Write end of node-function --
    sys.stdout.write(indent)
    sys.stdout.write('};\n')

## ---------- Generate list of BUILTIN functions --------##
builtins = []
argRegx =r'([^:]*):([^,]*),?'
def parseBuiltinFunction(match):
    name = match[0]
    if (name == 'Main' or name.startswith('Stencil')):
        pass # Ignore these functions
    else:
        # Parse input arguments
        argsm = re.findall(argRegx, match[1])
        args = []
        for arg in argsm:
            args.append((arg[0].strip(), arg[1].strip()))
        # Parse return type
        out = match[2].strip()
        if (out == 'Void'):
            out = ''
        # Add function to list of builtins
        builtins.append((name, args, out))

from subprocess import Popen, PIPE
# Run an empty program through the Equelle compiler to get the builtin functions
ec = Popen(['/equelle/build/compiler/ec','--input','-','--dump','symboltable'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
(ec_out,ec_err) = ec.communicate(input='')
if (len(ec_err)):
    exit('Equelle compiler error: '+ec_err)
else:
    funcRegx = r'-{18} Dump of function: ([^\s]+) -{18}\nFunction\(([^\)]*)\) -> ([^\n]*)'
    matches = re.findall(funcRegx, ec_out)
    for match in matches:
        parseBuiltinFunction(match)

## ---------- HINTING FUNCTION GENERATION START --------##
# Read the skeleton from hint.js
skel = open('hint.js','r')
for line in skel:
    ihs = line.find('##hint_start##');
    ihn = line.find('##hint_nodes##');
    iht = line.find('##hint_token_replacements##');
    ihb = line.find('##hint_builtin_functions##');
    # Insert the starting node, with indentation, so that it looks nice
    if (ihs >= 0):
        indent = ' '*ihs
        sys.stdout.write(indent)
        sys.stdout.write('var nstart = nodefun_')
        sys.stdout.write(start)
        sys.stdout.write(';\n')
    # Insert the hinting nodes into the skeleton
    elif (ihn >= 0):
        indent = ' '*ihn
        # Predeclare node functions
        nodefuns = ', '.join(['nodefun_'+node for node in supportedNodes.keys()])
        sys.stdout.write(indent)
        sys.stdout.write('var '+nodefuns+';\n')
        # Write all the node functions
        for node in supportedNodes.keys():
            writeNodeFunction(node, indent)
    # Insert token replacements
    elif (iht >= 0):
        separator = ' '
        indent = ' '*iht
        subindent = ' '*(iht+4)
        sys.stdout.write(indent)
        sys.stdout.write('var rep = {\n')
        # Write all the levels
        for token, replacement in tokenReplacements.iteritems():
            sys.stdout.write(subindent)
            sys.stdout.write(separator)
            sys.stdout.write(token)
            sys.stdout.write(': \'')
            sys.stdout.write(replacement)
            sys.stdout.write('\'\n')
            separator = ','
        sys.stdout.write(indent)
        sys.stdout.write('};\n')
    # Insert builtin functions
    elif (ihb >= 0):
        separator = ' '
        indent = ' '*ihb
        subindent = ' '*(ihb+4)
        sys.stdout.write(indent)
        sys.stdout.write('var builtins = {\n')
        for fun in builtins:
            subseparator = ''
            sys.stdout.write(subindent)
            sys.stdout.write(separator)
            # Function name
            sys.stdout.write('\''+fun[0]+'\': {')
            if len(fun[1])>0:
                subseparator = ','
                subsubseparator = ''
                # Function arguments
                sys.stdout.write(' args: [')
                for arg in fun[1]:
                    sys.stdout.write(subsubseparator)
                    sys.stdout.write('{ name: \'')
                    sys.stdout.write(arg[0])
                    sys.stdout.write('\', type: \'')
                    sys.stdout.write(arg[1])
                    sys.stdout.write('\'}')
                    subsubseparator = ','
                sys.stdout.write(']')
            if len(fun[2])>0:
                # Function output
                sys.stdout.write(subseparator)
                sys.stdout.write(' out: \'')
                sys.stdout.write(fun[2])
                sys.stdout.write('\' ')
            sys.stdout.write('}\n')
            separator = ','
        sys.stdout.write(indent)
        sys.stdout.write('};\n')
    # Write all other lines as they are
    else:
        sys.stdout.write(line)
