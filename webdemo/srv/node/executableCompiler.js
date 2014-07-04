/* Libraries */
var spawn = require('child_process').spawn,
    exec = require('child_process').exec,
    fs = require('fs-extra'),
    tmp = require('tmp'),
    _ = require('underscore');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');

// TODO: Everywhere, check that signatures is not empty strings or undefined!!!
/* Compilation function */
var compileExecutable = function(source, signature, conn, errorCleanup) {
    var sendProgress = function(p) { conn.sendJSON({ status: 'compiling', progress: p}) };
    /* Check that the c++ source was compiled by this server */
    helpers.signData(source, function(err, sign) {
        if (!sign || !signature || sign != signature) errorCleanup('Source signatures does not match');
        else {
            sendProgress(5);
            /* Create temporary make directory */
            tmp.dir({prefix: 'equelleCompileTmp'}, function(err, dir) {
                if (err) errorCleanup(err);
                else {
                    sendProgress(10);
                    /* Copy the make-skeleton directory contents to make directory */
                    var cp = exec('cp -R '+config.compiler_skel_dir+'/* '+dir, function(err, stdout, stderr) {
                        if (err) errorCleanup(stderr, dir);
                        else {
                            sendProgress(15);
                            /* Write the c++ code file to make directory */
                            fs.writeFile(dir+'/simulator.cpp', source, function(err) {
                                if (err) errorCleanup(err, dir);
                                else {
                                    sendProgress(20);
                                    /* Run cmake */
                                    var cmake = exec('cmake -DEquelle_DIR='+config.equelle_dir, {cwd: dir}, function(err, stdout, stderr) {
                                        if (err) errorCleanup(stderr, dir);
                                        else {
                                            sendProgress(40);
                                            /* Run make */
                                            var make = exec('make',{cwd: dir}, function(err, stdout, stderr) {
                                                if (err) errorCleanup(stderr, dir);
                                                else {
                                                    var progress = 90;
                                                    sendProgress(progress);
                                                    /* Sign, compress and send executable to client */
                                                    fs.readFile(dir+'/simulator', function(err, executable) {
                                                        if (err) errorCleanup(err, dir);
                                                        else {
                                                            var output = { sourceSign: sign };
                                                            /* Wait for both signature and compression to complete */
                                                            var done = _.after(2, function() {
                                                                if (output.err) errorCleanup(err, dir);
                                                                else {
                                                                    /* All done, send all results to client */
                                                                    sendProgress(100);
                                                                    conn.sendBytes(output.compressed);
                                                                    conn.sendJSON({ status: 'success', execSign: output.execSign, sourceSign: output.sourceSign });
                                                                    /* Clean up the temprary directory used for compilation */
                                                                    fs.remove(dir);
                                                                }
                                                            });
                                                            /* Sign the executable file */
                                                            helpers.signData(executable, function(err, sign) {
                                                                progress += 5;
                                                                sendProgress(progress);
                                                                if (err) output.err = err;
                                                                else output.execSign = sign;
                                                                done();
                                                            });
                                                            /* Compress the executable file to fit in the localStorage of a browser */
                                                            helpers.compressFile(dir+'/simulator', function(err, comp) {
                                                                progress += 5;
                                                                sendProgress(progress);
                                                                if (err) output.err = err;
                                                                else output.compressed = comp;
                                                                done();
                                                            });
                                                        }
                                                    });
                                                }
                                            });
                                        }
                                    });
                                }
                            });
                        }
                    });
                }
            });
        }
    });
};

/* The handleExecutableCompileConnection(connection) function */
module.exports = function(conn) {
    conn.sendJSON({ status: 'ready' });
    /* Error handling */
    var errorCleanup = function(err, dir) {
        console.log((new Date())+': Error during executable compilation:');
        console.log(err);
        /* Send error to client */
        conn.sendJSON({ status: 'failed', err: err.toString()});
        conn.close();
        /* Remove the temporary directory and all contents */
        if (dir) fs.remove(dir);
    };
    //TODO: handle aborts
    /* On receive compilation data from client */
    conn.on('message', function(mess) {
        console.log((new Date())+': Received executable compilation data from client');
        if (mess.type != 'utf8') errorCleanup('Not utf8 data');
        else try {
            var data = JSON.parse(mess.utf8Data);
            if (!data.source) throw('No source received');
            if (!data.sign) throw('No signature received');
            /* We should have all we need, start the executable compilation process */
            conn.sendJSON({ status: 'compiling', progress: 0});
            compileExecutable(data.source, data.sign, conn, errorCleanup);
        } catch (e) { errorCleanup(e) }
    });
}
