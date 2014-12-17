(function(){
    angular.module('eqksServer')
    /* Running of the executable on the server */
    .factory('eqksExecutableRun', ['$timeout','eqksPersistentSocket','eqksConfig','eqksInputs','eqksGrid','localStorageFile','parseSimulatorOutputData', function($timeout,socket,config,inputs,grid,lsFile,parser) {
        // Make a runner object that can emit events to listeners, this object will only be constructed once, and then reused every time a simulator is run
        var runner = _.clone(Backbone.Events);

        // Attach connection to server
        socket.attach(runner, 'ws://'+config.runHost+'/socket/', 'executable-run');

        // State object for retaining state of this runner between page navigations
        var state = {
            data: {},
            output: {
                stdout: '',
                stderr: '',
                data: {},
                progress: 0,
                packageOnComplete: false
            },
            running: false,
            progressClass: 'success'
        };

        /* Make parameters file and file list */
        var makeParametersAndFile = function(data) {
            var fileList = [];
            var paramArr = [], paramAdd = function(p) { paramArr.push(p+'\n') };
            // Grid dimension and sizes
            paramAdd('grid_dim='+data.grid.dimensions);
            switch (data.grid.dimensions) {
                case 3:
                paramAdd('nz='+data.grid.size[2]);
                paramAdd('dz='+data.grid.cellSize[2]);
                case 2:
                paramAdd('ny='+data.grid.size[1]);
                paramAdd('dy='+data.grid.cellSize[1]);
                case 1:
                paramAdd('nx='+data.grid.size[0]);
                paramAdd('dx='+data.grid.cellSize[0]);
            }
            paramAdd('abs_res_tol='+data.grid.abs_res_tol);
            // Input values
            _.each(data.singleScalars, function(input) {
                if (input.value) {
                    paramAdd(input.tag+'='+input.value);
                }
            });
            // Input files
            _.each(data.files, function(input) {
                if (input.value) {
                    /* The simulator will be running with ./outputs as working directory, so all files will be in ./../ */
                    paramAdd(input.tag+'_from_file=true');
                    paramAdd(input.tag+'_filename=../'+input.tag);
                    fileList.push({ name: input.tag, compressed: input.file.compressed, blob: input.file.data });
                }
            });
            // Write output to file
            paramAdd('output_to_file=true');
            // Set verbosity to 1(/2), so that we can make sure that the Newton solver is converging
            // .. if not, the program is probably doing something not very good, and we should kill it
            paramAdd('verbose=2');

            // Make the parameters into a blob
            var paramsBlob = new Blob(paramArr);

            // Add the executable and parameters to file list
            fileList.push({ name: config.executableName, compressed: data.executable.compressed, blob: data.executable.data });
            fileList.push({ name: config.paramFileName, compressed: false, blob: paramsBlob });

            return fileList;
        };

        /* Make simulator run-configuration object */
        var makeConfiguration = function(data, fileList) {
            return {
                name: config.executableName,
                paramFileName: config.paramFileName,
                signature: data.signature,
                files: _.map(fileList, function(f) { return { name: f.name, compressed: f.compressed } })
            }
        };

        /* Listen for output-data files */
        runner.on('socketData', function(data) {
            parser(data, function(err, out) {
                if (!err) {
                    // Create dataset container
                    if (!state.output.data[out.tag]) {
                        state.output.data[out.tag] = {
                            sets: [],
                            max: -Infinity,
                            min: Infinity
                        };
                    }
                    // Append new dataset
                    state.output.data[out.tag].sets[out.index] = {
                        data: out.data,
                        max: out.max,
                        min: out.min
                    };
                    // Re-calculate global max and min
                    state.output.data[out.tag].max = Math.max(state.output.data[out.tag].max, out.max);
                    state.output.data[out.tag].min = Math.min(state.output.data[out.tag].min, out.min);

                    runner.trigger('stateUpdate');
                }
            });
        });

        /* Listen for progress and completed messages */
        runner.on('socketMessage', function(msg) {
            if (msg.status == 'running') {
                state.running = true;
                // This is progress indication or ouput text
                if (msg.stdout) state.output.stdout += msg.stdout;
                if (msg.stderr) state.output.stderr += msg.stderr;
                if (msg.progress !== undefined) state.output.progress = msg.progress;
                state.progressClass = 'success';
                runner.trigger('stateUpdate');
            } else if (msg.status == 'sendingPackage') {
                // The server will be sending the output package next, handle this instead of output files
                runner.trigger('outputPackage', msg.name);
            } else if (msg.status == 'complete') {
                runner._done();
                state.running = false;
                state.output.progress = 100;
                state.progressClass = 'success';
                runner.trigger('stateUpdate');
                runner.trigger('complete');
            } else {
                state.running = false;
                state.progressClass = 'danger';
                runner.trigger('stateUpdate');
                runner.trigger('failed', 'Unknown server-status: '+msg.status);
            }
        });

        // Rethrow all socket errors to outside
        runner.on('socketError', function(error) {
            state.running = false;
            state.progressClass = 'danger';
            runner.trigger('stateUpdate');
            runner.trigger('failed', error);
        });


        /* Add functions for outside */
        runner.run = function() {
            if (state.running) return;

            /* Runs the currently stored executable file with currently stored grid and input files */
            runner.trigger('started');
            state.running = true;
            state.progressClass = 'success';

            // Make a copy of current grid and inputs
            state.data.grid = grid.get();
            state.data.singleScalars = inputs.getSingleScalars();
            state.data.files = inputs.getFiles();
            _.each(state.data.files, function(file) {
                file.file = lsFile.read(config.localStorageTags.inputFile+file.tag);
            });
            // Make copy of signatures and executable
            state.data.signature = localStorage.getItem(config.localStorageTags.executableSignature);
            state.data.executable = lsFile.read(config.localStorageTags.executable);

            // Clear old outputs
            state.output.stdout = '';
            state.output.stderr = '';
            state.output.data = {};
            state.output.progress = 0;
            $timeout(function() { runner.trigger('stateUpdate')}, 0);

            // Wait for server to become ready to compile
            runner.once('socketReady', function() {
                // Make parameters file and file list, then make run configuration
                var fileList = makeParametersAndFile(state.data);
                var configuration = makeConfiguration(state.data, fileList);
                configuration.packageOutput = state.output.packageOnComplete;

                runner._sendExpect({ command: 'config', config: configuration }, 'readyForFiles', function() {
                    // Server is now ready for all the files, send them IN ORDER!
                    _.each(fileList, function(file) {
                        runner._send(file.blob);
                    });

                    runner._expect('readyToRun', function() {
                        runner._send({ command: 'run' });
                    });
                });
            });

            // Connect to server for running
            runner._connect();
        };

        runner.abort = function() {
            runner._send({ command: 'abort' });
        };

        runner.setPackageOutput = function(pack) {
            runner._send({ command: 'setPackageOutput', pack: pack });
        };

        runner.getState = function() {
            return state;
        };

        return runner;
    }])
})();
