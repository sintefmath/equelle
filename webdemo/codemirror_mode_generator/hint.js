(function(){
    angular.module('equelleKitchenSinkEditorHints', [])
    /* Functions for the code-completion hints in the CodeMirror editor */
    .factory('equelleHints', function() { 
        /* We are going to be using the last element of an array a lot here, so add a little helper-function */
        if (!Array.prototype.last) { Array.prototype.last = function() { return (this.length ? this[this.length-1] : undefined); } };

        /* The hinting-nodes structure */
        ##hint_start##
        ##hint_nodes##
        ##hint_recursion_levels##

        /* The logic for finding the next token to use */
        var getFirstTokenAfter = function(currNode, pos, which, stack, nextStack) {
            /* Check that we do not exceed the allowed recursion level */
            var recursion = nrl[currNode] || nrl['default'];
            stack = stack || [];
            var count = _.reduce(stack, function(m,n) { return ((n == currNode) ? m+1 : m) }, 0);
            stack.push(currNode);

            var node = ns[currNode];
            var expects = [];
            var loopStart = 0, loopEnd = node.length;
            if (which !== undefined) { loopStart = which; loopEnd = which+1 }
            for (var i = loopStart; i < loopEnd; i++) {
                var expected = node[i]
                var nexts = _.clone(nextStack);
                if (expected.length > pos) {
                    var next = expected[pos];
                    /* Keep track of where we want to go after this token */
                    var o = { next: nexts };
                    /* If there are more tokens in this possibility, return to us */
                    if (expected.length > pos+1) {
                        nexts.push({ name: currNode, pos: pos+1, which: i});
                    }
                    /* Evaluate next token/node */
                    if (next.type == 'token' || next.type == 'literal') {
                        expects.push(_.extend(_.clone(o), next));
                    } else if (count <= recursion) {
                        /* If the recursion limit is broken, dont try to go deeper */
                        var children = getFirstTokenAfter(next.name, 0, undefined, stack, nexts);
                        _.each(children, function(next) {
                            if (!_.find(expects, function(expect) { return _.isEqual(expect,next) })) {
                                expects.push(next);
                            }
                        });
                    }
                }
                /* We also need to handle empty node posibilities */
                if (expected.length == 0) {
                    var p = nexts.pop();
                    var children = getFirstTokenAfter(p.name, p.pos, p.which, stack, nexts);
                    _.each(children, function(next) {
                        if (!_.find(expects, function(expect) { return _.isEqual(expect,next) })) {
                            expects.push(next);
                        }
                    });
                }
            }
            stack.pop();
            return expects;
        };

        var start = getFirstTokenAfter(nstart, 0, undefined, undefined, []);
        var getPossibleNextTokens = function(prevTokens) {
            /* Loop through the tokens that are already entered, and figure out where we are */
            var i = 0;
            var possibleNext = start;
            while (i < prevTokens.length && possibleNext.length) {
                var prev = prevTokens[i];
                var possibleNew = [];
                _.each(possibleNext, function(possible) {
                    if (possible.name == prev.name) {
                        var nexts = _.clone(possible.next);
                        var next = nexts.pop();
                        /* If there is a next step */
                        if (next) {
                            _.each(getFirstTokenAfter(next.name, next.pos, next.which, undefined, nexts), function(next) {
                                if (!_.find(possibleNew, function(pn) { return _.isEqual(pn, next) })) {
                                    possibleNew.push(next);
                                }
                            });
                        }
                    }
                });
                ++i;
                possibleNext = possibleNew;
            }
            /* possibleNext is now a list of a sort of code paths that we can follow */
            /* We are only interested in the different tokens we can input, so lets sort them out */
            var tokens = [];
            _.each(possibleNext, function(possibleToken) {
                if (!_.contains(tokens, possibleToken.name)) tokens.push(possibleToken.name);
            });
            //console.log(JSON.parse(JSON.stringify(possibleNext)));
            return tokens;
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
