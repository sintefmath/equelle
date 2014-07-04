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
 
// TODO: More thurough checkin that we are ready for each step

/* Execution function */
var handleExecute = function(conn, dir, errorCleanup) {
    var config, filesList, readyToRun = false;
    /* Function to run after we have received all files */
    var filesDone = function() {
        /* Verify the executable signature */
        helpers.signFile(dir+'/'+config.name, function(err,sign) {
            if (sign != config.signature) errorCleanup('The signatures did not match');
            else {
                var chmod = exec('chmod u+x '+dir+'/'+config.name, function(err, stdout, stderr) {
                    if (err) errorCleanup(stderr);
                    /* All the files are saved, the executable is verified and made runnable */
                    else {
                        readyToRun = true;
                        conn.sendJSON({ status: 'readyToRun' });
                    }
                });
            }
        });
    };
    /* Wait for all data to be read from streams in addition to exit */
    var allDone = _.after(3, function() {
        conn.sendJSON({ status: 'complete' });
        conn.close();
    });
    /* On receive data from client */
    conn.on('message', function(msg) {
        if (msg.type == 'utf8') {
            /* Command message */
            var data = JSON.parse(msg.utf8Data);
            switch (data.command) {
                /* This configuration tells us how the files are beeing sent, the signature of the executable, and some imporant filenames */
                case 'config':
                config = data.config;
                filesList = config.files.slice().reverse(); // Make a copy and reverse it, so we can use as stack later
                filesDone = _.after(filesList.length, filesDone);
                /* The client can now start to send the files */
                conn.sendJSON({ status: 'readyForFiles' });
                break;
                case 'run':
                if (!readyToRun) errorCleanup('Not ready to run yet');
                else {
                    conn.sendJSON({ status: 'running', progress: 0 });
                    /* Execute the simulator */
                    var process = spawn(dir+'/'+config.name, [config.paramFileName], { cwd: dir });
                    process.stdout.on('data', function(data) {
                        conn.sendJSON({ status: 'running', progress: 0, stdout: data.toString() });
                    });
                    process.stdout.on('end', allDone);
                    process.stderr.on('data', function(data) {
                        conn.sendJSON({ status: 'running', progress: 0, stderr: data.toString() });
                    });
                    process.stderr.on('end', allDone);
                    process.on('exit', function(code) {
                        conn.sendJSON({ status: 'running', progress: 100 });
                        allDone();
                    });

                } break;
                default: errorCleanup('Unexpected command: '+data.command);
            }
        } else {
            /* A file was sent */
            if (!filesList) errorCleanup('Not ready to receive files');
            else {
                var file;
                /* Get the next expected file */
                if (file = filesList.pop()) {
                    var path = dir+'/'+file.name;
                    if (file.compressed) {
                        /* Run through gzip uncompress */
                        helpers.decompressToFile(path, msg.binaryData, function(err) {
                            if (err) errorCleanup(err);
                            else filesDone();
                        });
                    } else {
                        /* Write directly to folder */
                        fs.writeFile(path, msg.binaryData, function(err) {
                            if (err) errorCleanup(err);
                            else filesDone();
                        });
                    }
                } else errorCleanup('Did not expect any more files');
            }
        }
    });
    /* Ready to start */
    conn.sendJSON({ status: 'readyForConfig' });
};

/* The handleExecutableRunConnection(connection) function */
module.exports = function(conn) {
    /* Error handling */
    var errorCleanup = function(err, dir) {
        console.log((new Date())+': Error during execution:');
        console.log(err);
        /* Send error to client */
        conn.sendJSON({ status: 'failed', err: err.toString()});
        conn.close();
        /* Remove the temporary directory and all contents */
        if (dir) fs.remove(dir);
    };
    /* Create a temporary directory for storing the executable files */
    tmp.dir({prefix: 'equelleRunTmp'}, function(err, dir) {
        if (err) errorCleanup(err);
        else {
            handleExecute(conn, dir, function(err) { errorCleanup(err,dir) });
        }
    });
}
