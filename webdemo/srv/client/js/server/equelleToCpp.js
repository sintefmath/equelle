(function(){
    angular.module('eqksServer')
    /* Compilation of Equelle code to C++ code by back-end */
    .factory('eqksEquelleToCpp', ['eqksPersistentSocket','eqksConfig','eqksInputs', function(socket,config,inputs) {
        // Make a compiler object that can emit events to listeners, this object will only be constructed once, and then reused every time a source code is compiled
        var compiler = _.clone(Backbone.Events);

        // Attach connection to server
        socket.attach(compiler, 'ws://'+config.compileHost+'/socket/', 'equelle-compile');

        // Rethrow all socket errors to outside
        compiler.on('socketError', function(error) {
            compiler.trigger('failed', error);
        });

        // Handle other messages than success
        compiler.on('socketMessage', function(msg) {
            if (msg.status == 'compileerror') {
                compiler._done();
                // Parse error messages
                var errors = [];
                _.each(msg.err.split('\n'), function(errmsg) {
                    var m = errmsg.match(/^(Parser|Lexer|Compile).*line (\d+):\s*(.*)$/);
                    if (m) {
                        errors.push({
                            line: parseInt(m[2])-1,
                            text: errmsg
                        });
                    }
                });
                compiler.trigger('compileerror', errors);
            } else {
                compiler.trigger('failed', 'Unknown server-status: '+msg.status);
            }
        });

        /* Add functions for outside */
        compiler.save = function(source) {
            // Stores the source code in the correct localStorage location
            localStorage.setItem(config.localStorageTags.equelleSource, source);
        };
        
        compiler.source = function() {
            // Reads the source code from the correct localStorage location
            return localStorage.getItem(config.localStorageTags.equelleSource);
        };

        compiler.hasCompiled = function() {
            // Check if we have a C++ source and a signature, which likely means that we have compiled something
            return (localStorage.getItem(config.localStorageTags.cppSource) && localStorage.getItem(config.localStorageTags.cppSourceSignature));
        };

        compiler.compile = function() {
            /* Compiles the currently stored Equelle-source to C++ code */
            compiler.trigger('started');

            // Clear old localStorage
            var source = localStorage.getItem(config.localStorageTags.equelleSource);
            localStorage.setItem(config.localStorageTags.cppSource, '');
            localStorage.setItem(config.localStorageTags.cppSourceSignature, '');

            // Wait for server to become ready to compile
            compiler.once('socketReady', function() {
                compiler._sendExpect({ source: source }, 'success', function(data) {
                    compiler._done();
                    // The compilation was successful
                    if (data.source && data.sign && data.inputs !== undefined) {
                        localStorage.setItem(config.localStorageTags.cppSource, data.source);
                        localStorage.setItem(config.localStorageTags.cppSourceSignature, data.sign);
                        inputs.parse(data.inputs);
                        compiler.trigger('completed');
                    } else {
                        compiler.trigger('failed', new Error('Not all expected results were returned'));
                    }
                });
            });

            // Connect to server for compilation
            compiler._connect();
        }

        // Return the compiler object
        return compiler;
    }])
})();
