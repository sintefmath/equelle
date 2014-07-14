(function(){
    angular.module('equelleKitchenSinkEditorHints', [])
    /* Functions for the code-completion hints in the CodeMirror editor */
    .factory('equelleHints', function() { 
        /* We are going to be using the last element of an array a lot here, so add a little helper-function */
        if (!Array.prototype.last) { Array.prototype.last = function() { return (this.length ? this[this.length-1] : undefined); } };

        /* Node-functions helpers */
        var combine = function(statements) {
            var ret = {};
            /* Extract available tokens for combination */
            _.each(statements, function(sf) {
                var r = sf();
                _.each(r, function(f, k) {
                    if(_.has(ret, k)) ret[k].push(f);
                    else ret[k] = [f];
                });
            });
            /* Check if there are multiple combinations for a token */
            _.each(ret, function(fa, t) {
                if (fa.length > 1) {
                    ret[t] = _.partial(combine, fa);
                } else {
                    ret[t] = fa[0];
                }
            });
            return ret;
        };

        /* The hinting-nodes structure */
        ##hint_nodes##
        ##hint_start##
        
        window.startfun = nstart;
        /* The logic for finding the next token to use */
        var empty = function() { return {} };
        window.emptyfun = empty;
        var getPossibleNextTokens = function(prevTokens) {
            var possibleNext = nstart(empty);
            /* Loop through the previous tokens and figure out what is possible next */
            var i = 0;
            while (i < prevTokens.length && _.keys(possibleNext).length) {
                var tn = prevTokens[i].name;
                if (_.has(possibleNext, tn)) possibleNext = possibleNext[tn]();
                else possibleNext = {};
                ++i;
            }
            return _.keys(possibleNext);
        };

        /* The function that takes a list of possible tokens to come next, and converts them into hint suggestions */
        ##hint_token_replacements##
        var convertPossibleTokensToSuggestions = function(tokens, current, line) {
            /* Create return object */
            var ret = {
                 list: []
                ,from: {line: line }
                ,to: {line: line }
            };
            /* Figure out where we are going to put the selected insert */
            if (current.type) {
                /* We want to replace whatever token is just before the cursor */
                ret.from.ch = current.start;
                ret.to.ch = current.end;
            } else {
                /* There is nothing before the cursor, place it after */
                ret.from.ch = current.end;
                ret.from.ch = current.end;
            }
            /* Build list */
            _.each(tokens, function(token) {
                var insert = token;
                if (rep[token]) insert = rep[token];
                ret.list.push({text: insert});
            });
            /* Return suggestions */
            return ret;
        };

        /* The function that decides what hints to show for a given location */
        var hint = function(cm, callback) {
            var pos = cm.getCursor();
            var currentToken = cm.getTokenAt(pos,true);
            /* If we are currently in a comment, do nothing */
            if (_.str.contains(currentToken.type,'EQUELLE-TOKEN-COMMENT')) return;
            /* ... the same goes for a line continuation */
            if (_.str.contains(currentToken.type,'EQUELLE-TOKEN-LINECONT')) return;
            
            /* Find out the text that is to the left of the cursor, so that we can figure out what this line of code says */
            /* ... keep in mind that the same line can be spread out by LINECONT, so we need to look at previous lines as well */
            var lines = [ cm.getRange({line: pos.line, ch: 0}, {line: pos.line, ch: pos.ch}) ];
            /* Work our way back to the firs non-continued line, and extract text as we go */
            var lastStateBefore;
            var i = pos.line-1;
            while (i >= 0) {
                var state = cm.getStateAfter(i, true);
                if (!state.continuedLine) {
                    lastStateBefore = state;
                    break;
                } else {
                    /* Extract all text until linecont token */
                    var line = cm.getLine(i);
                    var linecont = cm.getTokenAt({line: i, ch: line.length}, true);
                    lines.push(cm.getRange({line: i, ch: 0},{line: i, ch: linecont.start}));
                }
                i--;
            }
            /* Merge all lines to a single statement string */
            var str = lines.reverse().join('');
            var stateAfter = cm.getStateAfter(pos.line, true);

            /* Extract all tokens before the token just before the cursor in the current statement */
            var lastLineToken = stateAfter.lineTokens.last();
            var contLineNo = ((lastLineToken && lastLineToken.name == 'LINECONT') ? (stateAfter.continuedLineNo-1) : (stateAfter.continuedLineNo));
            var tokens = _.filter(stateAfter.lineTokens, function(token) { return (token.line < contLineNo || token.ch < currentToken.start) });

            /* We have now extracted the statement that comes before the cursor as both a string, and the parsed tokens */
            /* This should be enough information to make an educated guess of what the user might want to type in */
            var possibleTokens = getPossibleNextTokens(tokens);
            var suggestions = convertPossibleTokensToSuggestions(possibleTokens, currentToken, pos.line);
            callback(suggestions);
        };
        hint.async = true;
        /* Expose functions to outside */
        return {
             hint: hint
    }});
})();
