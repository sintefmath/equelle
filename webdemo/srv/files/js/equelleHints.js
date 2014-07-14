(function(){
    angular.module('equelleKitchenSinkEditorHints', [])
    /* Functions for the code-completion hints in the CodeMirror editor */
    .factory('equelleHints', function() { 
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
        var nodefun_f_decl_args, nodefun_comb_decl_assign, nodefun_assignment, nodefun_f_declaration, nodefun_f_type_expr, nodefun_f_starttype, nodefun_expr, nodefun_basic_type, nodefun_f_call_args, nodefun_number, nodefun_function_call, nodefun_stencil_statement, nodefun_loop_start, nodefun_type_expr, nodefun_statement, nodefun_declaration, nodefun_array, nodefun_f_startdef, nodefun_block, nodefun_stencil_access;
        nodefun_f_decl_args = function(p) {
            var end;
            var r1;
            var r1 = function() {
                var t1,t2;
                t1 = function() { return {',': t2} };
                t2 = _.partial(nodefun_declaration, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r1,p]);
            var s2,s1;
            var s2 = function() {
                return end();
            };
            var s1 = function() {
                var t1;
                t1 = _.partial(nodefun_declaration, function() { return end() });
                return t1();
            };
            return combine([s2,s1]);
        };
        nodefun_comb_decl_assign = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4,t5;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {':': t3} };
                t3 = _.partial(nodefun_type_expr, function() { return t4() });
                t4 = function() { return {'=': t5} };
                t5 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            return s1();
        };
        nodefun_assignment = function(p) {
            var end;
            end = p;
            var s2,s1;
            var s2 = function() {
                var t1,t2;
                t1 = _.partial(nodefun_f_startdef, function() { return t2() });
                t2 = _.partial(nodefun_block, function() { return end() });
                return t1();
            };
            var s1 = function() {
                var t1,t2,t3;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {'=': t3} };
                t3 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            return combine([s2,s1]);
        };
        nodefun_f_declaration = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {':': t3} };
                t3 = _.partial(nodefun_f_type_expr, function() { return end() });
                return t1();
            };
            return s1();
        };
        nodefun_f_type_expr = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4,t5,t6;
                t1 = _.partial(nodefun_f_starttype, function() { return t2() });
                t2 = function() { return {'(': t3} };
                t3 = _.partial(nodefun_f_decl_args, function() { return t4() });
                t4 = function() { return {')': t5} };
                t5 = function() { return {'RET': t6} };
                t6 = _.partial(nodefun_type_expr, function() { return end() });
                return t1();
            };
            return s1();
        };
        nodefun_f_starttype = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1;
                t1 = function() { return {'FUNCTION': end} };
                return t1();
            };
            return s1();
        };
        nodefun_expr = function(p) {
            var end;
            var r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13;
            var r4 = function() {
                var t1,t2;
                t1 = function() { return {'-': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r5 = function() {
                var t1,t2;
                t1 = function() { return {'+': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r6 = function() {
                var t1,t2;
                t1 = function() { return {'<': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r7 = function() {
                var t1,t2;
                t1 = function() { return {'>': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r12 = function() {
                var t1,t2,t3,t4,t5,t6;
                t1 = function() { return {'?': t2} };
                t2 = _.partial(nodefun_expr, function() { return t3() });
                t3 = function() { return {':': t4} };
                t4 = _.partial(nodefun_expr, function() { return t5() });
                t5 = function() { return {'%prec': t6} };
                t6 = function() { return {'?': end} };
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r1 = function() {
                var t1,t2,t3;
                t1 = function() { return {'[': t2} };
                t2 = function() { return {'INT': t3} };
                t3 = function() { return {']': end} };
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r2 = function() {
                var t1,t2;
                t1 = function() { return {'/': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r3 = function() {
                var t1,t2;
                t1 = function() { return {'*': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r14 = function() {
                var t1,t2;
                t1 = function() { return {'EXTEND': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r11 = function() {
                var t1,t2;
                t1 = function() { return {'NEQ': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r8 = function() {
                var t1,t2;
                t1 = function() { return {'LEQ': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r9 = function() {
                var t1,t2;
                t1 = function() { return {'GEQ': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r10 = function() {
                var t1,t2;
                t1 = function() { return {'EQ': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var r13 = function() {
                var t1,t2;
                t1 = function() { return {'ON': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r4,r5,r6,r7,r12,r1,r2,r3,r14,r11,r8,r9,r10,r13,p]);
            var s9,s8,s3,s2,s1,s7,s6,s5,s4;
            var s9 = function() {
                var t1;
                t1 = _.partial(nodefun_array, function() { return end() });
                return t1();
            };
            var s8 = function() {
                var t1;
                t1 = _.partial(nodefun_stencil_access, function() { return end() });
                return t1();
            };
            var s3 = function() {
                var t1,t2,t3;
                t1 = function() { return {'(': t2} };
                t2 = _.partial(nodefun_expr, function() { return t3() });
                t3 = function() { return {')': end} };
                return t1();
            };
            var s2 = function() {
                var t1;
                t1 = _.partial(nodefun_function_call, function() { return end() });
                return t1();
            };
            var s1 = function() {
                var t1;
                t1 = _.partial(nodefun_number, function() { return end() });
                return t1();
            };
            var s7 = function() {
                var t1;
                t1 = function() { return {'STRING_LITERAL': end} };
                return t1();
            };
            var s6 = function() {
                var t1;
                t1 = function() { return {'ID': end} };
                return t1();
            };
            var s5 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'-': t2} };
                t2 = _.partial(nodefun_expr, function() { return t3() });
                t3 = function() { return {'%prec': t4} };
                t4 = function() { return {'UMINUS': end} };
                return t1();
            };
            var s4 = function() {
                var t1,t2,t3;
                t1 = function() { return {'|': t2} };
                t2 = _.partial(nodefun_expr, function() { return t3() });
                t3 = function() { return {'|': end} };
                return t1();
            };
            return combine([s9,s8,s3,s2,s1,s7,s6,s5,s4]);
        };
        nodefun_basic_type = function(p) {
            var end;
            end = p;
            var s3,s2,s1,s7,s6,s5,s4;
            var s3 = function() {
                var t1;
                t1 = function() { return {'BOOL': end} };
                return t1();
            };
            var s2 = function() {
                var t1;
                t1 = function() { return {'VECTOR': end} };
                return t1();
            };
            var s1 = function() {
                var t1;
                t1 = function() { return {'SCALAR': end} };
                return t1();
            };
            var s7 = function() {
                var t1;
                t1 = function() { return {'VERTEX': end} };
                return t1();
            };
            var s6 = function() {
                var t1;
                t1 = function() { return {'EDGE': end} };
                return t1();
            };
            var s5 = function() {
                var t1;
                t1 = function() { return {'FACE': end} };
                return t1();
            };
            var s4 = function() {
                var t1;
                t1 = function() { return {'CELL': end} };
                return t1();
            };
            return combine([s3,s2,s1,s7,s6,s5,s4]);
        };
        nodefun_f_call_args = function(p) {
            var end;
            var r1;
            var r1 = function() {
                var t1,t2;
                t1 = function() { return {',': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            end = _.partial(combine, [r1,p]);
            var s2,s1;
            var s2 = function() {
                return end();
            };
            var s1 = function() {
                var t1;
                t1 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            return combine([s2,s1]);
        };
        nodefun_number = function(p) {
            var end;
            end = p;
            var s2,s1;
            var s2 = function() {
                var t1;
                t1 = function() { return {'FLOAT': end} };
                return t1();
            };
            var s1 = function() {
                var t1;
                t1 = function() { return {'INT': end} };
                return t1();
            };
            return combine([s2,s1]);
        };
        nodefun_function_call = function(p) {
            var end;
            end = p;
            var s2,s1;
            var s2 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {'(': t3} };
                t3 = _.partial(nodefun_f_call_args, function() { return t4() });
                t4 = function() { return {')': end} };
                return t1();
            };
            var s1 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'BUILTIN': t2} };
                t2 = function() { return {'(': t3} };
                t3 = _.partial(nodefun_f_call_args, function() { return t4() });
                t4 = function() { return {')': end} };
                return t1();
            };
            return combine([s2,s1]);
        };
        nodefun_stencil_statement = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4,t5;
                t1 = function() { return {'$': t2} };
                t2 = _.partial(nodefun_stencil_access, function() { return t3() });
                t3 = function() { return {'=': t4} };
                t4 = _.partial(nodefun_expr, function() { return t5() });
                t5 = function() { return {'$': end} };
                return t1();
            };
            return s1();
        };
        nodefun_loop_start = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'FOR': t2} };
                t2 = function() { return {'ID': t3} };
                t3 = function() { return {'IN': t4} };
                t4 = function() { return {'ID': end} };
                return t1();
            };
            return s1();
        };
        nodefun_type_expr = function(p) {
            var end;
            end = p;
            var s3,s2,s1,s7,s6,s5,s4;
            var s3 = function() {
                var t1,t2,t3,t4,t5;
                t1 = function() { return {'COLLECTION': t2} };
                t2 = function() { return {'OF': t3} };
                t3 = _.partial(nodefun_basic_type, function() { return t4() });
                t4 = function() { return {'ON': t5} };
                t5 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            var s2 = function() {
                var t1,t2,t3;
                t1 = function() { return {'COLLECTION': t2} };
                t2 = function() { return {'OF': t3} };
                t3 = _.partial(nodefun_basic_type, function() { return end() });
                return t1();
            };
            var s1 = function() {
                var t1;
                t1 = _.partial(nodefun_basic_type, function() { return end() });
                return t1();
            };
            var s7 = function() {
                var t1,t2;
                t1 = function() { return {'MUTABLE': t2} };
                t2 = _.partial(nodefun_type_expr, function() { return end() });
                return t1();
            };
            var s6 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'ARRAY': t2} };
                t2 = function() { return {'OF': t3} };
                t3 = function() { return {'INT': t4} };
                t4 = _.partial(nodefun_type_expr, function() { return end() });
                return t1();
            };
            var s5 = function() {
                var t1,t2,t3;
                t1 = function() { return {'SEQUENCE': t2} };
                t2 = function() { return {'OF': t3} };
                t3 = _.partial(nodefun_basic_type, function() { return end() });
                return t1();
            };
            var s4 = function() {
                var t1,t2,t3,t4,t5,t6;
                t1 = function() { return {'COLLECTION': t2} };
                t2 = function() { return {'OF': t3} };
                t3 = _.partial(nodefun_basic_type, function() { return t4() });
                t4 = function() { return {'SUBSET': t5} };
                t5 = function() { return {'OF': t6} };
                t6 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            return combine([s3,s2,s1,s7,s6,s5,s4]);
        };
        nodefun_statement = function(p) {
            var end;
            end = p;
            var s8,s3,s2,s1,s7,s6,s5,s4;
            var s8 = function() {
                var t1;
                t1 = _.partial(nodefun_stencil_statement, function() { return end() });
                return t1();
            };
            var s3 = function() {
                var t1;
                t1 = _.partial(nodefun_assignment, function() { return end() });
                return t1();
            };
            var s2 = function() {
                var t1;
                t1 = _.partial(nodefun_f_declaration, function() { return end() });
                return t1();
            };
            var s1 = function() {
                var t1;
                t1 = _.partial(nodefun_declaration, function() { return end() });
                return t1();
            };
            var s7 = function() {
                var t1,t2;
                t1 = _.partial(nodefun_loop_start, function() { return t2() });
                t2 = _.partial(nodefun_block, function() { return end() });
                return t1();
            };
            var s6 = function() {
                var t1,t2;
                t1 = function() { return {'RET': t2} };
                t2 = _.partial(nodefun_expr, function() { return end() });
                return t1();
            };
            var s5 = function() {
                var t1;
                t1 = _.partial(nodefun_function_call, function() { return end() });
                return t1();
            };
            var s4 = function() {
                var t1;
                t1 = _.partial(nodefun_comb_decl_assign, function() { return end() });
                return t1();
            };
            return combine([s8,s3,s2,s1,s7,s6,s5,s4]);
        };
        nodefun_declaration = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {':': t3} };
                t3 = _.partial(nodefun_type_expr, function() { return end() });
                return t1();
            };
            return s1();
        };
        nodefun_array = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3;
                t1 = function() { return {'[': t2} };
                t2 = _.partial(nodefun_f_call_args, function() { return t3() });
                t3 = function() { return {']': end} };
                return t1();
            };
            return s1();
        };
        nodefun_f_startdef = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4,t5;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {'(': t3} };
                t3 = _.partial(nodefun_f_call_args, function() { return t4() });
                t4 = function() { return {')': t5} };
                t5 = function() { return {'=': end} };
                return t1();
            };
            return s1();
        };
        nodefun_block = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'{': t2} };
                t2 = function() { return {'EOL': t3} };
                t3 = function() { return {'EOL': t4} };
                t4 = function() { return {'}': end} };
                return t1();
            };
            return s1();
        };
        nodefun_stencil_access = function(p) {
            var end;
            end = p;
            var s1;
            var s1 = function() {
                var t1,t2,t3,t4;
                t1 = function() { return {'ID': t2} };
                t2 = function() { return {'@': t3} };
                t3 = _.partial(nodefun_f_call_args, function() { return t4() });
                t4 = function() { return {'@': end} };
                return t1();
            };
            return s1();
        };
        var nstart = nodefun_statement;
        
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
        var rep = {
             SUBSET: 'Subset'
            ,EXTEND: 'Extend'
            ,VERTEX: 'Vertex'
            ,RET: '->'
            ,EOL: '\n'
            ,MUTABLE: 'Mutable'
            ,TRUE: 'True'
            ,NEQ: '!='
            ,GEQ: '>='
            ,CELL: 'Cell'
            ,EDGE: 'Edge'
            ,ARRAY: 'Array'
            ,FUNCTION: 'Function'
            ,XOR: 'Xor'
            ,FOR: 'For'
            ,SEQUENCE: 'Sequence'
            ,COLLECTION: 'Collection'
            ,SCALAR: 'Scalar'
            ,IN: 'In'
            ,EQ: '=='
            ,AND: 'And'
            ,ON: 'On'
            ,FALSE: 'False'
            ,OF: 'Of'
            ,FACE: 'Face'
            ,LEQ: '<='
            ,VECTOR: 'Vector'
            ,BOOL: 'Bool'
            ,NOT: 'Not'
            ,OR: 'Or'
        };
        var builtins = {
             'InteriorCells': { out: 'Collection Of Cell On InteriorCells() Subset Of AllCells()' }
            ,'BoundaryCells': { out: 'Collection Of Cell On BoundaryCells() Subset Of AllCells()' }
            ,'AllCells': { out: 'Collection Of Cell On AllCells() Subset Of AllCells()' }
            ,'InteriorFaces': { out: 'Collection Of Face On InteriorFaces() Subset Of AllFaces()' }
            ,'BoundaryFaces': { out: 'Collection Of Face On BoundaryFaces() Subset Of AllFaces()' }
            ,'AllFaces': { out: 'Collection Of Face On AllFaces() Subset Of AllFaces()' }
            ,'InteriorEdges': { out: 'Collection Of Edge On InteriorEdges() Subset Of AllEdges()' }
            ,'BoundaryEdges': { out: 'Collection Of Edge On BoundaryEdges() Subset Of AllEdges()' }
            ,'AllEdges': { out: 'Collection Of Edge On AllEdges() Subset Of AllEdges()' }
            ,'InteriorVertices': { out: 'Collection Of Vertex On InteriorVertices() Subset Of AllVertices()' }
            ,'BoundaryVertices': { out: 'Collection Of Vertex On BoundaryVertices() Subset Of AllVertices()' }
            ,'AllVertices': { out: 'Collection Of Vertex On AllVertices() Subset Of AllVertices()' }
            ,'FirstCell': { args: [{ name: 'faces', type: 'Collection Of Face'}], out: 'Collection Of Cell Subset Of AllCells()' }
            ,'SecondCell': { args: [{ name: 'faces', type: 'Collection Of Face'}], out: 'Collection Of Cell Subset Of AllCells()' }
            ,'IsEmpty': { args: [{ name: 'entities', type: '<multiple types possible>'}], out: 'Collection Of Bool' }
            ,'Centroid': { args: [{ name: 'entities', type: '<multiple types possible>'}], out: 'Collection Of Vector' }
            ,'Normal': { args: [{ name: 'faces', type: 'Collection Of Face'}], out: 'Collection Of Vector' }
            ,'InputScalarWithDefault': { args: [{ name: 'name', type: 'String'},{ name: 'default', type: 'Scalar'}], out: 'Scalar' }
            ,'InputCollectionOfScalar': { args: [{ name: 'name', type: 'String'},{ name: 'entities', type: '<multiple types possible>'}], out: 'Collection Of Scalar' }
            ,'InputDomainSubsetOf': { args: [{ name: 'name', type: 'String'},{ name: 'entities', type: '<multiple types possible>'}], out: 'Collection Of basicTypeString() error' }
            ,'InputSequenceOfScalar': { args: [{ name: 'name', type: 'String'}], out: 'Sequence Of Scalar' }
            ,'Divergence': { args: [{ name: 'values', type: 'Collection Of Scalar'}], out: 'Collection Of Scalar On AllCells()' }
            ,'Dot': { args: [{ name: 'v1', type: 'Collection Of Vector'},{ name: 'v2', type: 'Collection Of Vector'}], out: 'Collection Of Scalar' }
            ,'NewtonSolve': { args: [{ name: 'residual_function', type: '<multiple types possible>'},{ name: 'u_guess', type: 'Collection Of Scalar'}], out: 'Collection Of Scalar' }
            ,'NewtonSolveSystem': { args: [{ name: 'residual_function_array', type: '<multiple types possible>'},{ name: 'u_guess_array', type: 'Collection Of Scalar'}], out: 'Collection Of Scalar' }
            ,'Output': { args: [{ name: 'tag', type: 'String'},{ name: 'data', type: '<multiple types possible>'}]}
            ,'Sqrt': { args: [{ name: 's', type: 'Collection Of Scalar'}], out: 'Collection Of Scalar' }
            ,'MaxReduce': { args: [{ name: 'x', type: 'Collection Of Scalar'}], out: 'Scalar' }
            ,'MinReduce': { args: [{ name: 'x', type: 'Collection Of Scalar'}], out: 'Scalar' }
            ,'SumReduce': { args: [{ name: 'x', type: 'Collection Of Scalar'}], out: 'Scalar' }
            ,'ProdReduce': { args: [{ name: 'x', type: 'Collection Of Scalar'}], out: 'Scalar' }
        };
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
