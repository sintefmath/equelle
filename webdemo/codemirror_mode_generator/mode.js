(function(mod) {
    if (typeof exports == 'object' && typeof module == 'object') // CommonJS
        mod(require('../../lib/codemirror'));
    else if (typeof define == 'function' && define.amd) // AMD
        define(['../../lib/codemirror'], mod);
    else // Plain browser env
        mod(CodeMirror);
})(function(CodeMirror) {
    'use strict';

    /* The lexer */
    var lexer = ( function() {
        var r = [
            ##lexer_regexes##
        ];
        var a = [
            ##lexer_actions##
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

    /* The parser, keeps track of indenations */
    var parser = ( function() {
        /* The actual parsing function */
        return function(token, state) {
        };
    })();

    /* The style function, which gives a token a css-class based on its name */
    var tokenStyle = function(token) {
        if (token && token.name) switch(token.name) {
            ##style_cases##
            default: return token.name;
        }
        else return null;
    };

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
