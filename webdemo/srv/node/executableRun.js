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

/* The handleExecutableRunConnection(connection) function */
module.exports = function(conn) {
    var errorCleanup = function(err, dir) {
        console.log((new Date())+': Error during execution:');
        console.log(err);
        // Send error to client
        conn.sendJSON({ status: 'failed', err: err.toString()});
        // TODO: Delete temporary directory
    };
    /* Create a temporary directory for storing the executable files */
    tmp.dir({prefix: 'equelleRunTmp'}, function(err, dir) {
        if (err) errorCleanup(err);
        else {
            console.log('Running executable in folder: '+dir);
            /* Ready to receive files from client */
            conn.sendJSON({ status: 'readyForConfig' });
            /* On receive compilation data from client */
            var uploadConfig;
            var uploadFiles;
            var filesDone;
            conn.on('message', function(mess) {
                if (mess.type == 'utf8') {
                    /* Command message */
                    var data = JSON.parse(mess.utf8Data);
                    switch (data.command) {
                        case 'config':
                        /* This tells us how the rest of the message will be sent */
                        uploadConfig = data.config;
                        uploadFiles = uploadConfig.fileList.slice().reverse(); // Reverse so we can pop off later
                        filesDone = _.after(uploadFiles.length, function() {
                            // When all files are uploaded, verify the simulator executable
                            helpers.signFile(dir+'/simulator', function(err,sign) {
                                if (sign != uploadConfig.sign) errorCleanup('The signatures did not match');
                                else {
                                    console.log('Marking exec as executable');
                                    var chmod = spawn('chmod', ['u+x', dir+'/simulator']);
                                    chmod.on('exit', function(code) {
                                        if (code != 0) errorCleanup('Could not chmod file');
                                        else conn.sendJSON({ status: 'readyToRun' })
                                    });
                                }
                            });
                        });
                        console.log('Got configuration:'); console.log(uploadConfig);
                        conn.sendJSON({ status: 'readyForFiles' });
                        break;
                        case 'run':
                        /* Start the running of the provided program */
                        console.log('RUNNING EXECUTABLE');
                        var process = spawn(dir+'/simulator', ['params.param'], { cwd: dir });
                        helpers.readStdOutErr(process, function(out,err) {
                            console.log('Out: '); console.log(out);
                            console.log('Err: '); console.log(err);
                        });
                        break;
                        case 'abort':
                        /* Abort the running program somehow */
                        // TODO: Implement this
                        break;
                        default: errorCleanup('Unexpected command: '+data.command);
                    }
                } else {
                    var file;
                    if (file = uploadFiles.pop()) {
                        var path = dir+'/'+file.name;
                        if (file.compressed) {
                            /* Run through gzip uncompress */
                            console.log((new Date())+': Got executable bytes from client');
                            console.log('Compressed length: '+mess.binaryData.length);
                            //console.log('First 10 : '+mess.binaryData.slice(0,10));
                            //console.log('Last 10  : '+mess.binaryData.slice(-10));
                            console.log('First 10 : '); console.log(mess.binaryData.slice(0,10));
                            console.log('last  10 : '); console.log(mess.binaryData.slice(-10));
                            helpers.decompressToFile(path, mess.binaryData, function(err) {
                                //if (err) errorCleanup('Could not decompress file: "'+path+'"');
                                if (err) errorCleanup(err);
                                else filesDone();
                            });
                        } else {
                            /* Write directly to folder */
                            console.log('Got binary file data:');
                            console.log(mess.binaryData.length);
                            fs.writeFile(path, mess.binaryData, function(err) {
                                //if (err) errorCleanup('Could not write file: "'+path+'"');
                                if (err) errorCleanup(err);
                                else filesDone();
                            });
                        }
                    } else errorCleanup('Did not expect any more files');
                }
            });
        }
    });
    conn.on('close', function(code) { console.log('Connection closed: '+code); });
    conn.on('error', function(err) { console.log('Connection error: '+err); });
}
