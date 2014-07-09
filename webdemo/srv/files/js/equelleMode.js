(function(mod) {
    if (typeof exports == 'object' && typeof module == 'object') // CommonJS
        mod(require('../../lib/codemirror'));
    else if (typeof define == 'function' && define.amd) // AMD
        define(['../../lib/codemirror'], mod);
    else // Plain browser env
        mod(CodeMirror);
})(function(CodeMirror) {
    'use strict';

    /* ----- START AUTO GENERATED LEXER ----- */
    var lexer = ( function() {
        var r = [
             /^\"(\\.|[^\\"])*\"/
            ,/^Collection/
            ,/^Sequence/
            ,/^Array/
            ,/^Of/
            ,/^On/
            ,/^Extend/
            ,/^Subset/
            ,/^Scalar/
            ,/^Vector/
            ,/^Bool/
            ,/^Cell/
            ,/^Face/
            ,/^Edge/
            ,/^Vertex/
            ,/^Function/
            ,/^And/
            ,/^Or/
            ,/^Not/
            ,/^Xor/
            ,/^True/
            ,/^False/
            ,/^For/
            ,/^In/
            ,/^Mutable/
            ,/^[A-Z][0-9a-zA-Z_]*/
            ,/^[a-z][0-9a-zA-Z_]*/
            ,/^[0-9]+/
            ,/^[0-9]+[.][0-9]+/
            ,/^#.*/
            ,/^<=/
            ,/^>=/
            ,/^==/
            ,/^!=/
            ,/^->/
            ,/^[$@:=()+\-*/^<>{},|?\[\]]/
            ,/^$/
            ,/^[.][.][.][\t ]*$/
            ,/^[\t ]+/
            ,/^[0-9]+[0-9a-zA-Z_]+/
            ,/^[0-9]+[.][0-9]+[0-9a-zA-Z_]+/
            ,/^./
        ];
        var a = [
             function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'STRING_LITERAL'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'COLLECTION'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'SEQUENCE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'ARRAY'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'OF'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'ON'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'EXTEND'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'SUBSET'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'SCALAR'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'VECTOR'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'BOOL'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'CELL'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'FACE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'EDGE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'VERTEX'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'FUNCTION'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'AND'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'OR'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'NOT'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'XOR'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'TRUE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'FALSE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'FOR'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'IN'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'MUTABLE'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'BUILTIN'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'ID'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'INT'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'FLOAT'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.text = yytext; r.name = 'COMMENT'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'LEQ'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'GEQ'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'EQ'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'NEQ'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'RET'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = yytext[0]; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; r.name = 'EOL'; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; throw ("Lexer error on line " + yylineno + ": this is not a number \'" + yytext + "\'"); return undefined; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; throw ("Lexer error on line " + yylineno + ": this is not a number \'" + yytext + "\'"); return undefined; return r; }
            ,function(yytext,yylineno) { var r = {}; r.match = yytext; throw ("Lexer error on line " + yylineno + ": unexpected character \'" + yytext + "\'"); return undefined; return r; }
        ];
        /* The actual lexing function */
        return function(stream, lineno) {
            /* Find all matches */
            var matches = [], lengths = [];
            for (var i = 0; i < r.length; i++) {
                var m = stream.match(r[i],false);
                if (m) {
                    matches.push(i);
                    lengths.push(m[0].length);
                }
            }
            /* Select the first longest match */
            var max = -1, maxI;
            for (var i = 0; i < matches.length; i++) {
                if (lengths[i] > max) {
                    max = lengths[i];
                    maxI = matches[i];
                }
            }
            /* If match found, use that */
            if (maxI !== undefined) {
                var m = stream.match(r[maxI]);
                return (a[maxI])(m[0],lineno);
            }
            /* None found, return nothing, this shouldn't really happen */
            return undefined;
        };
    })();
    /* ----- END AUTO GENERATED LEXER ----- */

    /* ----- START AUTO GENERATED PARSER ----- */
    var parser = ( function() {
        /* The actual parsing function */
        return function(token, state) {
            //TODO: Do something cool here
        };
    })();
    /* ----- END AUTO GENERATED PARSER ----- */

    /* ----- START AUTO GENERATED TOKEN STYLER ----- */
    var tokenStyle = function(token) {
        if (token && token.name) switch(token.name) {
            case "COMMENT":
            return "comment";
    
            case "STRING_LITERAL":
            return "string";
    
            case "OF":
            case "ON":
            case "EXTEND":
            case "SUBSET":
            case "MUTABLE":
            case "AND":
            case "OR":
            case "NOT":
            case "XOR":
            case "FOR":
            case "IN":
            case "$":
            case "@":
            return "keyword";
    
            case "INT":
            case "FLOAT":
            return "number";
    
            case "LEQ":
            case "GEQ":
            case "EQ":
            case "NEQ":
            case ":":
            case "=":
            case "+":
            case "-":
            case "/":
            case "^":
            case "<":
            case ">":
            case "?":
            return "operator";
    
            case "BUILTIN":
            case "FUNCTION":
            return "builtin";
    
            case "(":
            case ")":
            case "[":
            case "]":
            case "{":
            case "}":
            case "|":
            return "bracket";
    
            case "TRUE":
            case "FALSE":
            return "atom";
    
            case "ID":
            return "variable";
    
            case "COLLECTION":
            case "SEQUENCE":
            case "ARRAY":
            case "SCALAR":
            case "VECTOR":
            case "BOOL":
            case "CELL":
            case "FACE":
            case "EDGE":
            case "VERTEX":
            return "type";
    
    
            default: return token.name;
        }
        else return null;
    };
    /* ----- END AUTO GENERATED TOKEN STYLER ----- */

    /* Some configuration is needed if the Equelle language is changed */
    var  LINECONT_STRING = '...'
        ,EOL_TOKEN = 'EOL';
    // TODO: REMEMBER THIS!

    /* The actual mode logic */
    CodeMirror.defineMode('equelle', function() {
        return {
            startState: function() { return {
            }}
            ,token: function token(s, state) {
                /* Run the lexer, this will return a token, and advance the stream */
                var lex;
                try { lex = lexer(s,0); }
                /* This throws error for unexpected characters, the stream has advanced past them, so mark them as error */
                catch (e) { return 'error'; }

                /* If no error, we should have gotten a token back, send this token to the parser */
                if (lex && lex.name) parser(lex);

                /* If we are at the end-of-line, we should also make sure that the EOL token is sent to the parser, unless the last token was a LINECONT (...) */
                if (s.eol() && lex && lex.name != EOL_TOKEN && lex.match && !_.str.startsWith(lex.match,LINECONT_STRING)) token(s,state);

                /* We are now ready to return the style of the last token we read */
                return tokenStyle(lex);
            }
        };
    });
});
