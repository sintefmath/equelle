(function(){
    angular.module('equelleKitchenSinkEquelleCompile', ['equelleKitchenSinkConfiguration'])
    /* Classes that compile Equelle code to C++ code, and parses required inputs from the Equelle source */
    .factory('equelleCompiler', ['eqksConfig', function(eqksConfig) { 
        /* A class that abstracts away the logic of compiling Equelle source code on the server, extends Events from Backbone */
        Compiler = function() { };
        /* Extend with event emitter */
        _.extend(Compiler.prototype, Backbone.Events);
        /* The function that does the actual compilation of the Equelle code */
        Compiler.prototype.compile = function(source) {
            self = this;
            var triggerEvent = function() { self.trigger.apply(self, arguments) };
            /* Clear data in browser from old compilations */
            localStorage.eqksSource = source;
            localStorage.eqksCompiled = '';
            localStorage.eqksCompiledSign = '';
            /* Indicate that we have started the process */
            triggerEvent('started');
            /* Open connection to server */
            var sock = new WebSocket('ws://'+eqksConfig.compileHost+'/socket/', 'equelle-compile');
            /* Handle socket errors */
            sock.onerror = function(err) { triggerEvent('failed', err) };
            /* Message protocol */
            var errorTriggered = false;
            sock.onmessage = function(msg) {
                try {
                    var data = JSON.parse(msg.data);
                    switch (data.status) {
                        /* Ready to receive source code */
                        case 'ready':
                        sock.send(JSON.stringify({source: source}));
                        break;
                        /* Compilation was successful, c++ code and signature is attached */
                        case 'success':
                        console.log(data);
                        localStorage.eqksCompiled = data.source;
                        localStorage.eqksCompiledSign = data.sign;
                        sock.close();
                        break;
                        /* A compiler error, this is different from a failure, and errors are shown to user */
                        case 'compilerror':
                        triggerEvent('compileerror', data.err);
                        errorTriggered = true;
                        sock.close();
                        break;
                        case 'failed': throw data.err; break;
                        default: throw ('Unrecognized server status: '+data.status);
                    }
                } catch (e) { triggerEvent('failed', e); errorTriggered = true; sock.close() }
            }
            /* Once socket is closed, check that everything went smoothly */
            sock.onclose = function() {
                if (!errorTriggered) {
                    if (!localStorage.eqksCompiled || !localStorage.eqksCompiledSign) {
                        triggerEvent('failed', 'Not all expected results were found in localStorage');
                    } else {
                        triggerEvent('completed');
                    }
                }
            };
        };

        /* The function that parses the Equelle code and saves required (and optional) inputs to the localStorage */
        var parseInputsFromSource = function() {
            /* Scan the source for input values */
            var inputs = [], match;
            //var funcregx = /^\s*([a-z]\w*)\s*=\s*Input([^\(\s]*)\(\s*"(\w*)"(,\s*(.*))?\)/gm;
            // TODO: Document this!
            var funcregx = /^\s*([a-z]\w*)\s*(?:\:[^\=\n]*\s*)?=\s*Input([^\(\s]*)\(\s*"(\w*)"(?:,\s*(.*))?\)/gm;
            if (localStorage.eqksSource) while (match = funcregx.exec(localStorage.eqksSource)) {
                funcregx.lastIndex = match.index+match[0].length;
                /* Parse matched Regular Expression */
                var input = {
                    name: match[1],
                    tag: match[3]
                };
                /* Determine type of input */
                switch (match[2]) {
                    case 'ScalarWithDefault': input.type = 'scalar'; input.domain = 'single'; input.default = match[4]; break;
                    case 'CollectionOfScalar': input.type = 'scalar'; break;
                    case 'DomainSubsetOf': input.type = 'indices'; break;
                    case 'SequenceOfScalar': input.type = 'scalarsequence'; input.domain = 'single'; break;
                    default: throw 'Cannot determine type of input with tag: "'+input.tag+'"';
                }
                /* Determine domain of input */
                if (!input.domain) switch (match[4]) {
                    case 'InteriorCells()': input.domain = 'cells'; input.subset = 'interior'; break;
                    case 'BoundaryCells()': input.domain = 'cells'; input.subset = 'boundary'; break;
                    case 'AllCells()': input.domain = 'cells'; input.subset = 'all'; break;
                    case 'InteriorFaces()': input.domain = 'faces'; input.subset = 'interior'; break;
                    case 'BoundaryFaces()': input.domain = 'faces'; input.subset = 'boundary'; break;
                    case 'AllFaces()': input.domain = 'faces'; input.subset = 'all'; break;
                    case 'InteriorEdges()': input.domain = 'edges'; input.subset = 'interior'; break;
                    case 'BoundaryEdges()': input.domain = 'edges'; input.subset = 'boundary'; break;
                    case 'AllEdges()': input.domain = 'edges'; input.subset = 'all'; break;
                    case 'InteriorVertices()': input.domain = 'vertices'; input.subset = 'interior'; break;
                    case 'BoundaryVertices()': input.domain = 'vertices'; input.subset = 'boundary'; break;
                    case 'AllVertices()': input.domain = 'vertices'; input.subset = 'all'; break;
                    default:
                    var found;
                    if (found = _.find(inputs, function(input) { return input.name == match[4]; })) {
                        input.domain = found.domain;
                        input.subset = 'var:'+found.name;
                    } else throw 'Cannot determine domain of input with tag: "'+input.tag+'"';
                }
                inputs.push(input);
            }
            localStorage.eqksInputs = JSON.stringify(inputs);
        };

        /* Expose class to outside */
        return {
             Compiler: Compiler
            ,hasCompiled: function() { return (localStorage.eqksCompiled && localStorage.eqksCompiledSign); }
            ,parseInputs: parseInputsFromSource
            ,getInputs: function() { if (localStorage.eqksInputs) return JSON.parse(localStorage.eqksInputs); else return []; }
    }}]);
})();
