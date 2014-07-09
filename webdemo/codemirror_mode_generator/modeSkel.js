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
    ##lexer##
    /* ----- END AUTO GENERATED LEXER ----- */

    /* ----- START AUTO GENERATED PARSER ----- */
    ##parser##
    /* ----- END AUTO GENERATED PARSER ----- */

    /* ----- START AUTO GENERATED TOKEN STYLER ----- */
    ##styler##
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
