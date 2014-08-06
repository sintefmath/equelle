/* Libraries */
var exec = require('child_process').exec,
    fs = require('fs-extra'),
    tmp = require('tmp'),
    _ = require('underscore');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');


/* Compilaction procedure */
var compileExecutable = function(state, source, signature, conn, quit) {
    var tryAsync = helpers.tryAsync('C++ compiler', quit);
    var sendProgress = function(p) { conn.sendJSON({ status: 'compiling', progress: p}) };

    var checkSignature, createTempDir, copySkel, writeCPP, cmake, make, signCompress, sendResults;

    /* Check that the c++ source was compiled by this server */
    checkSignature = function() {
        if (state.abort) return;

        tryAsync(helpers.signData, source)
        .complete(function(sign) {
            if (!sign || !signature || sign !== signature) quit('Source signatures does not match');
            else {
                sendProgress(5);
                createTempDir();
            }
        })
        .run();
    };

    /* Create temporary make directory */
    createTempDir = function() {
        if (state.abort) return;

        tryAsync(tmp.dir, { prefix: 'equelleCompileTmp' })
        .complete(function(dir) {
            state.dir = dir;
            sendProgress(10);
            copySkel();
        })
        .run();
    };

    /* Copy the make-skeleton directory contents to make directory */
    var copySkel = function() {
        if (state.abort) return;

        state.currentExec = tryAsync(false, exec, 'cp -R '+config.compiler_skel_dir+'/* '+state.dir)
        .complete(function() {
            sendProgress(15);
            writeCPP();
        })
        .error(function(err, stdout, stderr) {
            quit(stderr);
        })
        .always(function() {
            state.currentExec = null;
        })
        .run();
    };

    /* Write the c++ code file to make directory */
    writeCPP = function() {
        if (state.abort) return;

        tryAsync(fs.writeFile, state.dir+'/simulator.cpp', source)
        .complete(function() {
            sendProgress(20);
            cmake();
        })
        .run();
    };

    /* Run cmake */
    cmake = function() {
        if (state.abort) return;

        state.currentExec = tryAsync(false, exec, 'cmake -DCMAKE_BUILD_TYPE=Release -DEquelle_DIR='+config.equelle_dir, { cwd: state.dir })
        .complete(function() {
            sendProgress(40);
            make();
        })
        .error(function(err, stdout, stderr) {
            quit(stderr);
        })
        .always(function() {
            state.currentExec = null;
        })
        .run();
    };

    /* Run make */
    make = function() {
        if (state.abort) return;

        state.currentExec = tryAsync(false, exec, 'make', { cwd: state.dir })
        .complete(function() {
            sendProgress(90);
            signCompress();
        })
        .error(function(err, stdout, stderr) {
            quit(stderr);
        })
        .always(function() {
            state.currentExec = null;
        })
        .run();
    }
    
    var results = {};
    /* Sign and compress the executable file */
    signCompress = function() {
        if (state.abort) return;

        var done = _.after(2, function() {
            sendProgress(100);
            sendResults()
        });

        // Attach source signature to results
        results.sourceSign = signature;

        // Read file for signing
        tryAsync(fs.readFile, state.dir+'/simulator')
        .complete(function(executable) {
            if (state.abort) return;

            // Sign the file
            tryAsync(helpers.signData, executable)
            .complete(function(sign) {
                // Attach executable signature to results
                results.execSign = sign;
            })
            .always(done)
            .run();
        })
        .error(done)
        .run();

        // Compress the executable file
        tryAsync(helpers.compressFile, state.dir+'/simulator')
        .complete(function(compressed) {
            // Attach executable file contents to results
            results.compressed = compressed;
        })
        .always(done)
        .run();
    }

    /* Send the results of the compilation to the client */
    sendResults = function() {
        if (state.abort) return;

        if (results.sourceSign && results.execSign && results.compressed) {
            // First send the executable byte data
            conn.sendBytes(results.compressed);
            // Then send completed status, togethere with signatures
            conn.sendJSON({ status: 'success', execSign: results.execSign, sourceSign: results.sourceSign });
            // Lastly, cleanup the temporary directory
            fs.remove(state.dir);
        } else {
            helpers.logError('C++ compiler', 'Not all expected results were attached');
            quit('An error occured during compilation');
        }
    };

    // Start function
    checkSignature();
};

/* The handleExecutableCompileConnection(connection) function */
module.exports = function(conn) {
    var state = {};
    var quit = function(error) { 
        // Send error to client
        conn.sendJSON({ status: 'failed', err: err.toString()});
        conn.close();
        // Remove the temporary directory and all contents
        if (state.dir) fs.remove(state.dir);
    };

    /* Handle abortions of current compilation process */
    var abort = function(reason) {
        // Set abort flag
        state.abort = true;

        // If a long-running exec is currently doing something, try to stop it
        if (state.currentExec) try {
            state.currentExec.kill();
        } catch (e) {}

        // Close connection
        quit('Compilation aborted, reason: '+reason);
    };

    /* On reveive compilation commands from the client */
    conn.on('message', function(msg) {
        if (msg.type != 'utf8') errorCleanup('Not utf8 data');
        else try {
            var data = JSON.parse(msg.utf8Data);
            switch (data.command) {
                case 'compile':
                    // Received a compilation command, check that we are not already compiling something
                    if (state.compiling) {
                        abort('Already compiling');
                    } else {
                        state.compiling = true;
                        // Compile an executable file
                        if (!data.source) throw('No source received');
                        if (!data.sign) throw('No signature received');
                        // We should have all we need, start the executable compilation process
                        conn.sendJSON({ status: 'compiling', progress: 0});
                        compileExecutable(state, data.source, data.sign, conn, quit);
                    }
                break;
                case 'abort':
                    abort('Client abort');
                break;
                default:
                    quit('Could not understand the command');
            }
        } catch (error) {
            quit(error);
        }
    });

    /* Let the client know we are ready to do something */
    conn.sendJSON({ status: 'ready' });
}
