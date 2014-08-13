/* Libraries */
var spawn = require('child_process').spawn,
    exec = require('child_process').exec,
    fs = require('fs-extra'),
    tmp = require('tmp'),
    _ = require('underscore'),
    str = require('string');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');
 

/* Returns a function that can be called periodically to look for any new files since last call, and send them to the specified connection */
var createFileChecker = function(state, conn, tryAsync) {
    // Keep track of already sent files
    var sentFiles = [];
    return function() {
        // Do a read of the directory's contents
        tryAsync(false, fs.readdir, state.outDir)
        .complete(function(files) {
            // Find new files
            _.each(files, function(file) {
                if (!_.contains(sentFiles, file)) {
                    sentFiles.push(file);
                    // Found a new file, send it
                    tryAsync(false, fs.readFile, state.outDir+'/'+file, { encoding: 'utf8' })
                    .complete(function(data) {
                        var dataBuffer = helpers.packageSimulatorOutput(file, data);
                        conn.sendBytes(dataBuffer);
                    })
                    .run();
                }
            });
        })
        .run();
    };
};

var progressRegx = /^progress =\s*(\S*)\s*\n/;
var convergeRegx = /^Newton solver converged/;
/* Execution function */
var handleExecute = function(state, conn, quit, handleAnother) {
    var tryAsync = helpers.tryAsync('Simulation run', quit);
    var expectCommand = helpers.expectCommand(conn, quit);

    var getConfig, getFiles, verifySignature, chmodExecutable, waitForRun, runComplete, sendOutputPackage, close;

    /* We first expect a config command, with configuration data */
    var execConfig, filesList;
    getConfig = function() {
        expectCommand('config', function(data) {
            state.config = execConfig = data.config;
            // Make a copy of the list of files that will be sent, and reverse it, so we can us it as a stack later
            filesList = execConfig.files.slice().reverse();
            
            getFiles();
        });

        conn.sendJSON({ status: 'ready' });
    };

    /* Next, we expect to receive the binary data of all files indicated */
    getFiles = function() {
        var receiveFile;
        var done = _.after(filesList.length, function() {
            conn.removeListener('message', receiveFile);
            verifySignature();
        });

        receiveFile = function(msg) {
            if (msg.type != 'binary') {
                quit('Waiting for a binary message, but got text');
            } else {
                // We got one of the files, should be the next of the stack
                var file = filesList.pop();
                var path = state.dir+'/'+file.name;
                // Write the file to temporary directory
                if (file.compressed) {
                    // The file is compressed, run through gzip
                    tryAsync(helpers.decompressToFile, path, msg.binaryData)
                    .always(done)
                    .run();
                } else {
                    // The file is not compressed, write directly to disk
                    tryAsync(fs.writeFile, path, msg.binaryData)
                    .always(done)
                    .run();
                }
            }
        };

        conn.on('message', receiveFile);

        /* The client can now start to send the files */
        conn.sendJSON({ status: 'readyForFiles' });
    };

    /* Verify the signature of the executable file with our secret key, so that we don't run anything that is not created by this server */
    verifySignature = function() {
        tryAsync(helpers.signFile, state.dir+'/'+execConfig.name)
        .complete(function(sign) {
            if (!sign || !execConfig.signature || sign !== execConfig.signature) {
                quit('The signatures did not match');
            } else {
                chmodExecutable();
            }
        })
        .run();
    };

    /* chmod the simluator file so that we can execute it */
    chmodExecutable = function() {
        tryAsync(false, exec, 'chmod u+x '+state.dir+'/'+execConfig.name)
        .complete(function() {
            waitForRun();
        })
        .error(function(err, stdout, stderr) {
            quit(stderr);
        })
        .run();
    };

    /* Now, we are ready to run the executable, wait for the client to initiate */
    var watcherPromise, watcher;
    var checkFiles = createFileChecker(state, conn, tryAsync);
    waitForRun = function() {
        expectCommand('run', function(data) {
            conn.sendJSON({ status: 'running', progress: 0 });

            /* Setup the listener which checks for new files periodically */
            watcher = fs.watch(state.outDir, { persistent: false }, function(event) {
                // persistent: false, means we don't care about changes after node is killed
                // TODO: This function is documented as not beeing reliable, is there any way around it?
                // To prohibit reading files in the middle of a write, we wait for a little time after each file-event in the output directory
                clearTimeout(watcherPromise);
                watcherPromise = setTimeout(checkFiles, 500);
            });

            /* Execute the simulator */
            var process = state.simulatorProcess = spawn(state.dir+'/'+execConfig.name, ['../'+execConfig.paramFileName], { cwd: state.outDir });
            var stdout = '', line = '';
            var lastConverge = Date.now(); //TODO: Use this as an indicator for a stalled simulator

            /* Upon execution, we wait for process to exit, and for both stdout and stderr to drain, before we send completed event */
            var done = _.after(3, runComplete);

            /* Read data on stdout */
            process.stdout.on('data', function(data) {
                stdout += data.toString();
                // Extract lines from stdout
                var i;
                while ((i = stdout.indexOf('\n')) > 0) {
                    line = stdout.substr(0,i+1);
                    stdout = stdout.substr(i+1);
                    /* Parse line */
                    // Look for a progress indication
                    var pm = line.match(progressRegx);
                    if (pm) try {
                        var progress = parseInt(pm[1]);
                        // Make sure we are in the 0-100 range
                        progress = Math.max(0,Math.min(100,progress));
                        conn.sendJSON({ status: 'running', progress: progress });
                    } catch (e) {}
                    // Send all other lines to client
                    else {
                        // If line contains solver convergence, use it to keep track of stalling processes
                        var cm = line.match(convergeRegx);
                        if (cm) lastConverge = Date.now();

                        // Send stdout
                        conn.sendJSON({ status: 'running', stdout: line });
                    }
                }
            });
            
            /* Read data on stderr */
            process.stderr.on('data', function(data) {
                conn.sendJSON({ status: 'running', stderr: data.toString() });
            });

            /* Wait for process to exit */
            process.stdout.on('end', done);
            process.stderr.on('end', done);
            process.on('exit', function(code) {
                conn.sendJSON({ status: 'running', progress: 100 });
                done();
            });
        });

        conn.sendJSON({ status: 'readyToRun' });
    };

    /* The simulator has run to completion, possibly with errors. All stdout/err data has been sent to client */
    runComplete = function() {
        /* Stop the file watcher and the promise to check for files later, check for new files immediately */
        watcher.close();
        clearTimeout(watcherPromise); 
        checkFiles();


        /* Check wether we want to package the output on completion */
        if (state.config.packageOutput) {
            sendOutputPackage();
        } else {
            close();
        }

    };

    /* The client wants a package of all the data */
    sendOutputPackage = function() {
        tryAsync(helpers.compressDirectory, state.outDir)
        .complete(function(compressed) {
            // Create a temporary file to write to
            tryAsync(tmp.file, { prefix: 'equelleOutputPackage' })
            .complete(function(path, fd) {
                // Write to temporary file so we can download later
                tryAsync(fs.write, fd, compressed, 0, compressed.length, 0)
                .complete(function() {
                    // Close file
                    tryAsync(fs.close, fd)
                    .complete(function() {
                        var name = path.substr(path.lastIndexOf('equelleOutputPackage')+20);
                        conn.sendJSON({ status: 'sendingPackage', name: name });
                    })
                    .always(close)
                    .run();
                })
                .error(close)
                .run();
            })
            .error(close)
            .run();
        })
        .error(close)
        .run();
    };

    /* Everything is done, do cleanup */
    close = function() {
        fs.remove(state.dir);

        /* Send completed event to client, and close connection */
        conn.sendJSON({ status: 'complete' });

        handleAnother()
    }

    // Start function
    getConfig();
};


/* The handleExecutableRunConnection(connection) function */
module.exports = function(handlerName, domain, conn, handleAnother) {
    var state = {};
    var quit = function(error) { 
        // Send error to client
        conn.sendJSON({ status: 'failed', err: error.toString()});
        conn.close();
        // Remove the temporary directory and all contents
        if (state.dir) fs.remove(state.dir);
    };
    var tryAsync = helpers.tryAsync('Simulation run', quit);

    /* Handle abortions of current simulation process */
    var abort = function(reason) {
        // Set the abort flag
        state.abort = true;
        // If the simulator was started, try to stop it
        if (state.simulatorProcess) try {
            helpers.killAll(state.simulatorProcess);
        } catch (error) {}

        // Close connection
        quit('Simulator aborted');
    };
    conn.on('message', function(msg) {
        if (msg.type == 'utf8') try {
            var data = JSON.parse(msg.utf8Data);
            if (data.command && data.command == 'abort') {
                // We got an abort message from the client
                abort('Client abort');
            } else if (data.command && data.command == 'setPackageOutput') {
                // Set the flag to send output package on completion
                if (state.config) state.config.packageOutput = !!data.pack;
            }
        } catch (error) {}
    });


    /* Create a temporary directory for storing the executable files */
    tryAsync(tmp.dir, { prefix: 'equelleRunTmp' })
    .complete(function(dir) {
        state.dir = dir;
        /* Create output directory */
        var outDir = dir+'/outputs';
        tryAsync(fs.mkdir, outDir)
        .complete(function() {
            state.outDir = outDir;
            // We are ready to handle execution commands from the client
            handleExecute(state, conn, quit, handleAnother);
        })
        .run();
    })
    .run();

    /* Return the abortion function */
    return abort;
}

