(function(){
    angular.module('eqksEquelleHints', [])
    /* Functions for the code-completion hints in the CodeMirror editor */
    .factory('equelleCodemirrorHints', function() { 
        /* Node-functions helpers */
        var combine = function(statements) {
            var ret = {};
            /* Extract available tokens for combination */
            _.each(statements, function(sf) {
                var r = sf();
                _.each(r, function(f, k) {
                    if (_.has(ret, k)) ret[k].push(f);
                    else ret[k] = [f];
                });
            });
            /* Check if there are multiple combinations for a token */
            _.each(ret, function(fa, t) {
                if (fa.length > 1) ret[t] = _.partial(combine, fa);
                else ret[t] = fa[0];
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

        /* Suggestion helpers */
        var createBuiltinHint = function(name, opts) {
            var ret = {};
            /* Create insertion text */
            ret.text = name+'(';
            var argns = _.map(opts.args, function(arg) { return arg.name});
            ret.text += argns.join(', ');
            ret.text += ')'
            /* Create display text */
            ret.displayText = 'Function: '+name+'()';
            /* Set where we want to go after the completion */
            if (argns.length > 0) {
                /* Set cursor after first argument name */
                ret.goto = name.length+1+opts.args[0].name.length;
            } else {
                /* Set cursor after ending parenthesis */
                ret.goto = name.length+2;
            }
            return ret;
        };
        var createFunctionHint = function(fobj) {
            var ret = {};
            /* Create insertion text */
            ret.text = fobj.name+'(';
            ret.text += fobj.args.join(', ');
            ret.text += ')'
            /* Create display text */
            ret.displayText = 'Function: '+fobj.name+'()';
            /* Set where we want to go after the completion */
            if (fobj.args.length > 0) {
                /* Set cursor after first argument name */
                ret.goto = fobj.name.length+1+fobj.args[0].length;
            } else {
                /* Set cursor after ending parenthesis */
                ret.goto = name.length+2;
            }
            return ret;
        };

        var filterHints = function(list, current) {
            /* Remove the suggestions that doesn't match what is already written in front of the cursor */
            if (_.str.isBlank(current.string)) return list;
            return _.filter(list, function(s) {
                return (_.str.startsWith(s.text, current.string));
            });
        };

        /* The function that takes a list of possible tokens to come next, and converts them into hint suggestions */
        ##hint_token_replacements##
        ##hint_builtin_functions##
        var convertPossibleTokensToSuggestions = function(tokens, current, line, state) {
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
                ret.to.ch = current.end;
            }
            /* Build list */
            var showUserFuncs = false;
            if (_.contains(tokens, 'ID')) {
                /* Always show variables first */
                tokens = _.without(tokens, 'ID');
                _.each(state.blockVariables, function(varr) {
                    _.each(varr, function(v) {
                        ret.list.push({text: v+' ', displayText: 'Variable: '+v});
                    });
                });
                showUserFuncs = true;
            }
            if (_.contains(tokens, 'STRING_LITERAL')) {
                /* Then show string */
                tokens = _.without(tokens, 'STRING_LITERAL');
                ret.list.push({text: '""', displayText: 'String: ""', goto: 1});
            }
            if (_.contains(tokens, 'INT')) {
                /* Then show integer */
                tokens = _.without(tokens, 'INT');
                ret.list.push({text: '', displayText: 'Int: 1', goto: 1});
            }
            if (_.contains(tokens, 'FLOAT')) {
                /* Then show floating point */
                tokens = _.without(tokens, 'FLOAT');
                ret.list.push({text: '', displayText: 'Float: 1.0', goto: 1});
            }
            /* Then show other tokens */
            var showBuiltins = _.contains(tokens, 'BUILTIN');
            tokens = _.without(tokens, 'BUILTIN');
            _.each(tokens, function(token) {
                /* Insert actual token */
                var insert = token;
                if (rep[token]) insert = rep[token];
                /* Also insert a space after a token */
                insert += ' ';
                ret.list.push({text: insert, displayText: 'Keyword: '+insert});
            });
            if (showUserFuncs) {
                /* Then show user functions */
                _.each(state.blockFunctions, function(farr) {
                    _.each(farr, function(f) {
                        ret.list.push(createFunctionHint(f));
                    });
                });
            }
            if (showBuiltins) {
                /* Then show builtin functions */
                _.each(builtins, function(o, n) {
                    ret.list.push(createBuiltinHint(n,o));
                });
            }
            /* Return filtered suggestions */
            ret.list = filterHints(ret.list, current);
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
            var lastLineToken = _.last(stateAfter.lineTokens);
            var contLineNo = ((lastLineToken && lastLineToken.name == 'LINECONT') ? (stateAfter.continuedLineNo-1) : (stateAfter.continuedLineNo));
            var tokens = _.filter(stateAfter.lineTokens, function(token) { return (token.line < contLineNo || token.ch < currentToken.start) });

            /* We have now extracted the statement that comes before the cursor as both a string, and the parsed tokens */
            /* This should be enough information to make an educated guess of what the user might want to type in */
            var possibleTokens = getPossibleNextTokens(tokens);
            var suggestions = convertPossibleTokensToSuggestions(possibleTokens, currentToken, pos.line, stateAfter);

            /* Hook to completion events */
            CodeMirror.on(suggestions, 'pick', function(completion) {
                if (_.has(completion,'goto')) {
                    /* Goto a specific position after the completion is done */
                    var pos = suggestions.from;
                    pos.ch += completion.goto;
                    cm.setCursor(pos);
                }
            });

            /* Show the widget to select a completion */
            callback(suggestions);
        };
        hint.async = true;
        /* Expose functions to outside */
        return {
             hint: hint
    }});
})();
