(function(){
    angular.module('equelleKitchenSinkExecutableCompile', ['equelleKitchenSinkConfiguration', 'equelleKitchenSinkHelpers'])
    /* Classes that compile C++ source to an executable file, and handles the storage of required (and optional) inputs */
    .factory('executableCompiler', ['eqksConfig', 'localStorageFile', function(eqksConfig, localStorageFile) { 
        /* A class that abstracts away the logic of compiling Equelle source code on the server, extends Events from Backbone */
        Compiler = function() { };
        /* Extend with event emitter */
        _.extend(Compiler.prototype, Backbone.Events);
        /* The function that does the actual compilation of the C++ code */
        Compiler.prototype.compile = function() {
            self = this;
            var triggerEvent = function() { self.trigger.apply(self, arguments) };
            /* Clear data in browser from old compilations */
            localStorageFile.remove('eqksExecutable');
            localStorage.eqksExecutableSign = '';
            localStorage.eqksExecutableCompiledSign = '';
            /* Indicate work beeing done to user */
            triggerEvent('started');
            /* Open connection to server */
            var sock = new WebSocket('ws://'+eqksConfig.runHost+'/socket/', 'executable-compile');
            /* Handle socket errors */
            sock.onerror = function(err) { triggerEvent('failed', err) };
            /* Wait for both socket to close, and executable file to be written to localStorage before we check the output */
            var errorTriggered = false;
            var checkResult = _.after(2, function() {
                if (!errorTriggered) {
                    if (!localStorageFile.hasFile('eqksExecutable') || !localStorage.eqksExecutableSign || !localStorage.eqksExecutableCompiledSign) {
                        triggerEvent('failed', 'Not all expected results were found in localStorage');
                    } else {
                        triggerEvent('completed');
                    }
                }
            });
            /* Message protocol */
            sock.onmessage = function(msg) {
                if (msg.data instanceof Blob || msg.data instanceof ArrayBuffer) {
                    /* This should be the actual executable file that we get in return */
                    if (msg.data instanceof ArrayBuffer) msg.data = new Blob([msg.data]);
                    /* Save data in localStorage */
                    localStorageFile.write('eqksExecutable', msg.data, function() { checkResult() });
                } else {
                    try {
                        var data = JSON.parse(msg.data);
                        switch (data.status) {
                            /* Ready for us to send the source code */
                            case 'ready':
                            sock.send(JSON.stringify({source: localStorage.eqksCompiled, sign: localStorage.eqksCompiledSign}));
                            break;
                            /* Compilation progress sent underways */
                            case 'compiling':
                            triggerEvent('progress', data.progress);
                            break;
                            /* The compilation was a success, the signatures are attached, and the actual executable will be sent by itself */
                            case 'success':
                            localStorage.eqksExecutableSign = data.execSign;
                            localStorage.eqksExecutableCompiledSign = data.sourceSign;
                            sock.close();
                            break;
                            case 'failed': throw data.err; break;
                            default: throw ('Unrecognized server status: '+data.status);
                        }
                    } catch (e) { triggerEvent('failed', e); errorTriggered = true; sock.close() }
                }
            };
            /* Socket is closed */
            sock.onclose = function() { checkResult() };
        };


        /* Expose class to outside */
        return {
             Compiler: Compiler
            ,hasCompiled: function() { return (localStorageFile.hasFile('eqksExecutable') && localStorage.eqksCompiledSign == localStorage.eqksExecutableCompiledSign ); }
    }}]);
})();
