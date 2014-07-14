#!/usr/bin/python
# ---- This script generates code-completion hints, based on the Bison configuration file ----
import re
import fileinput
import sys
import shlex

# Bison grammar rules
nodes = {}
currentNode = None
nodeStartRegx = r'([^:]*):\s*((\'{|[^{])*)'
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
bisonfile = open('../../compiler/equelle_parser.y')
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
flexfile = open('../../compiler/equelle_lexer.l')
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
nodes['expr'] = [ "ID", "number", "STRING_LITERAL", "function_call" ]

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


#import pprint
#pprint.pprint(supportedNodes)

#print(supportedNodes[start])

# Write out node-function
def writeNodeFunction(node, indent):
    # Write function start
    sys.stdout.write(indent)
    sys.stdout.write(node)
    sys.stdout.write(' = function(p) {\n')
    # Write all possibilites from this node
    subindent = indent+(' '*4)
    subsubindent = subindent+(' '*4)
    possibilities = []
    possCount = 1
    for possibility in supportedNodes[node]:
        name = 'p'+str(possCount)
        possibilities.append(name)
        possCount += 1
        # Write possibility start
        sys.stdout.write(subindent+'var '+name+' = function() {\n')
        # Predeclare all tokens in this possibility
        sys.stdout.write(subsubindent)
        sys.stdout.write('var ')
        sys.stdout.write(','.join(['t'+str(n+1) for n in range(len(possibility))]))
        sys.stdout.write(';\n')
        # Write actual tokens
        tokCount = 1
        for token in possibility:
            # Write token start
            sys.stdout.write(subsubindent)
            sys.stdout.write('t')
            sys.stdout.write(str(tokCount))
            sys.stdout.write(' = ')
            # Figure out next token
            next = 'p'
            if (tokCount < len(possibility)):
                next = 't'+str(tokCount+1)
            # Write actual token
            if (token['type'] == 'node'):
                sys.stdout.write('_.partial(')
                sys.stdout.write(token['name'])
                sys.stdout.write(', ')
                sys.stdout.write(next)
                sys.stdout.write(')')
            else: #token['type'] == 'token'
                sys.stdout.write('function() { return { \'')
                sys.stdout.write(token['name'])
                sys.stdout.write('\': ')
                sys.stdout.write(next)
                sys.stdout.write(' }}')
            # Write token end
            sys.stdout.write(';\n')
            tokCount += 1
        # Write possibility end
        sys.stdout.write(subindent)
        sys.stdout.write('}\n')
    # Return combination of possibilities
    sys.stdout.write(subindent)
    sys.stdout.write('return combine([')
    sys.stdout.write(','.join(possibilities))
    sys.stdout.write(']);\n')
    # Write function end
    sys.stdout.write(indent)
    sys.stdout.write('};\n')


writeNodeFunction(start,'')
sys.exit('Done')
## ---------- HINTING FUNCTION GENERATION START --------##
# Read the skeleton from hint.js
skel = open('hint.js','r')
for line in skel:
    ihs = line.find('##hint_start##');
    ihn = line.find('##hint_nodes##');
    ihr = line.find('##hint_recursion_levels##');
    iht = line.find('##hint_token_replacements##');
    # Insert the starting node, with indentation, so that it looks nice
    if (ihs >= 0):
        indent = ' '*ihs
        sys.stdout.write(indent)
        sys.stdout.write('var nstart = \'')
        sys.stdout.write(start)
        sys.stdout.write('\';\n')
    # Insert the hinting nodes into the skeleton
    elif (ihn >= 0):
        separator = ' '
        indent = ' '*ihn
        subindent = ' '*(ihn+4)
        sys.stdout.write(indent)
        sys.stdout.write('var ns = {\n')
        # Write all nodes
        for name,expects in supportedNodes.iteritems():
            subseparator = ''
            sys.stdout.write(subindent)
            sys.stdout.write(separator)
            sys.stdout.write(name)
            sys.stdout.write(': [')
            # Write all expected children
            ## TODO: Do we care about literals?
            for expect in expects:
                subsubseparator = ''
                sys.stdout.write(subseparator)
                sys.stdout.write('[')
                for token in expect:
                    sys.stdout.write(subsubseparator)
                    sys.stdout.write('{type:\'')
                    sys.stdout.write(token['type'])
                    sys.stdout.write('\',')
                    if (token['type'] == 'literal'):
                        sys.stdout.write('text:\'')
                        sys.stdout.write(token['text'])
                    else:
                        sys.stdout.write('name:\'')
                        sys.stdout.write(token['name'])
                    sys.stdout.write('\'}')
                    subsubseparator = ','
                sys.stdout.write(']')
                subseparator = ','
            sys.stdout.write(']\n')
            separator = ','
        sys.stdout.write(indent)
        sys.stdout.write('};\n')
    # Insert recursion levels
    elif (ihr >= 0):
        separator = ' '
        indent = ' '*ihr
        subindent = ' '*(ihr+4)
        sys.stdout.write(indent)
        sys.stdout.write('var nrl = {\n')
        # Write all the levels
        for name, level in recursion.iteritems():
            sys.stdout.write(subindent)
            sys.stdout.write(separator)
            sys.stdout.write(name)
            sys.stdout.write(': ')
            sys.stdout.write(str(level))
            sys.stdout.write('\n')
            separator = ','
        sys.stdout.write(indent)
        sys.stdout.write('};\n')
    # Insert recursion levels
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
    # Write all other lines as they are
    else:
        sys.stdout.write(line)
