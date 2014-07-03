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

        /* Expose class to outside */
        return {
             Compiler: Compiler
            ,hasCompiled: function() { return (localStorage.eqksCompiled && localStorage.eqksCompiledSign); }
    }}]);
})();
