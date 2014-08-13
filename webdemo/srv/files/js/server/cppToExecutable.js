(function(){
    angular.module('eqksServer')
    /* Compilation of Equelle code to C++ code by back-end */
    .factory('eqksCppToExecutable', ['eqksPersistentSocket','eqksConfig','eqksInputs','localStorageFile', function(socket,config,inputs,lsFile) {
        // Make a compiler object that can emit events to listeners, this object will only be constructed once, and then reused every time a source code is compiled
        var compiler = _.clone(Backbone.Events);

        // Attach connection to server
        socket.attach(compiler, 'ws://'+config.compileHost+'/socket/', 'executable-compile');

        // Rethrow all socket errors to outside
        compiler.on('socketError', function(error) {
            compiler.trigger('failed', error);
            console.log('ERROR:', error);
        });

        // Update button status
        var state = {
            buttonProperties: undefined,
            buttonScope: undefined,
            status: 'ready',
            progress: 0
        };
        var updateButton = function(apply) {
            if (state.buttonProperties && state.buttonScope) {
                if (apply) {
                    state.buttonScope.$apply(function() { updateButton(false) });
                } else {
                    switch (state.status) {
                        case 'ready':
                            state.buttonProperties.working = false;
                            state.buttonProperties.class = 'primary';
                            state.buttonProperties.showProgress = false;
                            state.buttonProperties.progress = 0;
                            break;
                        case 'working':
                            state.buttonProperties.working = true;
                            state.buttonProperties.class = 'primary';
                            state.buttonProperties.showProgress = true;
                            state.buttonProperties.progress = state.progress;
                            break;
                        case 'failed':
                            state.buttonProperties.working = false;
                            state.buttonProperties.class = 'danger';
                            state.buttonProperties.showProgress = false;
                            state.buttonProperties.progress = 0;
                            break;
                        case 'done':
                            state.buttonProperties.working = false;
                            state.buttonProperties.class = 'success';
                            state.buttonProperties.showProgress = false;
                            state.buttonProperties.progress = 0;
                            break;
                    }
                }
            }
        };

        // Handle other messages than success
        compiler.on('socketMessage', function(msg) {
            if (msg.status == 'compiling') {
                // This is progress indication, send to button
                state.status = 'working';
                state.progress = msg.progress;
                updateButton(true);
            } else {
                state.status = 'failed';
                state.progress = 0;
                updateButton(true);
                compiler.trigger('failed', 'Unknown server-status: '+msg.status);
            }
        });

        /* Add functions for outside */
        compiler.hasCompiled = function() {
            // Check if we have an executable and signatures, which likely means that we have compiled something
            return ( lsFile.hasFile(config.localStorageTags.executable) &&
                     localStorage.getItem(config.localStorageTags.executableSignature) &&
                     localStorage.getItem(config.localStorageTags.executableSourceSignature) == localStorage.getItem(config.localStorageTags.cppSourceSignature) );
        };

        compiler.bindButtonProperties = function(properties, $scope) {
            state.buttonProperties = properties;
            state.buttonScope = $scope;
            updateButton();

            $scope.$on('$destroy', function() {
                state.buttonProperties = undefined;
                state.buttonScope = undefined;
            });
        };

        compiler.compile = function() {
            /* Compiles the currently stored Equelle-source to C++ code */
            compiler.trigger('started');

            // Update button
            state.status = 'working';
            state.progress = 0;
            updateButton();

            // Clear old localStorage
            var cppSource = localStorage.getItem(config.localStorageTags.cppSource);
            var cppSign = localStorage.getItem(config.localStorageTags.cppSourceSignature);

            lsFile.remove(config.localStorageTags.executable);
            localStorage.setItem(config.localStorageTags.executableSignature, '');
            localStorage.setItem(config.localStorageTags.executableSourceSignature, '');

            // Wait for server to become ready to compile
            compiler.once('socketReady', function() {
                var executable;
                compiler.once('socketData', function(data) {
                    executable = data;
                });

                compiler._sendExpect({ command: 'compile', source: cppSource, sign: cppSign }, 'success', function(data) {
                    compiler._done();
                    // The compilation was successful
                    if (data.execSign && data.sourceSign == localStorage.getItem(config.localStorageTags.cppSourceSignature) && executable) {
                        lsFile.write(config.localStorageTags.executable, executable, true, function(err) {
                            if (!err) {
                                // The executable is written to localStorage, save the rest
                                localStorage.setItem(config.localStorageTags.executableSignature, data.execSign);
                                localStorage.setItem(config.localStorageTags.executableSourceSignature, data.sourceSign);
                                compiler.trigger('completed');

                                // Update button
                                state.status = 'done';
                                state.progress = 100;
                                updateButton(true);
                            } else {
                                // Update button
                                state.status = 'failed';
                                state.progress = 0;
                                updateButton(true);

                                compiler.trigger('failed', new Error('Could not write executable to localStorage'));
                            }
                        });
                    } else {
                        // Update button
                        state.status = 'failed';
                        state.progress = 0;
                        updateButton(true);

                        compiler.trigger('failed', new Error('Not all expected results were returned'));
                    }
                    
                });
            });

            // Connect to server for compilation
            compiler._connect();
        }

        return compiler;
    }])
})();
