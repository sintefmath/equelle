/* Libraries */
var _ = require('underscore'),
    spawn = require('child_process').spawn,
    exec = require('child_process').exec,
    fs = require('fs-extra'),
    psTree = require('ps-tree');
/* Own modules */
var config = require('./config.js');


/* Helper function module */
(function(module) {
    /* Generates a Message Authentication Code from buffer using the secret key, then calls waiting function */
    module.signData = function(data, outCB) {
        // Spawn OpenSSL and send the data
        var signer = exec('openssl dgst -sha512 -hmac "'+config.secret_key+'"', function(err, stdout, stderr) {
            // Retreive signature or errors
            if (err) outCB(stderr, stdout);
            else outCB(null, stdout.slice(9,-1));
        });
        signer.stdin.end(data);
    };

    /* Reads data from a file, generates MAC, then calls waiting function */
    module.signFile = function(file, outCB) {
        fs.readFile(file, function(err, data) {
            if (err) outCB(err);
            else module.signData(data, outCB);
        });
    }

    /* Makes a compressed version (file.compressed) of file using gzip, then calls waiting functino with the file data as a buffer */
    module.compressFile = function(file, outCB) {
        // Spawn gzip which reads the file itself, and sends output to file.compressed
        // TODO: Read data directly from stdout instead?
        var compress = exec('gzip -c -9 '+file+' > '+file+'.compressed', function(err, stdout, stderr) {
            if (err) outCB(stderr);
            else {
                // Read the compressed file into a buffer, and call waiting function
                fs.readFile(file+'.compressed', function(err, compressed) {
                    if (err) outCB(err);
                    else outCB(null, compressed);
                });
            }
        });
    };

    /* Decompresses a buffer into a file, calls waiting function when done */
    module.decompressToFile = function(file, data, outCB) {
        // Spawn gzip which writes decompressed data to file, and send buffer to stdin
        var decompress = exec('gzip -d - > '+file, function(err, stdout, stderr) {
            if (err) outCB(stderr);
            else outCB(null);
        });
        decompress.stdin.end(data);
    };

    /* Logging helpers */
    module.logInfo = function(serverName, info) {
        console.log((new Date())+': '+serverName+': '+info);
    };
    module.logError = function(serverName, error) {
        console.log((new Date())+': '+serverName+' error: '+error);
    };

    /* Error handler helper */
    module.tryAsync = function(serverName, quitCB) {
        return function(quit, func) {
            // Check if the quit argument is supplied, and extract remainder of arguments
            var args, thisVal = this;
            if (typeof quit == 'function') {
                args = Array.prototype.slice.call(arguments, 1);
                func = quit;
                quit = false; // Default to false
            } else {
                args = Array.prototype.slice.call(arguments, 2);
            }
            // Create the return-object
            var completeCB, alwaysCB, errorCB;
            var retObj = {
                 complete: function(callback) {
                     completeCB = callback;
                     return retObj;
                 }
                ,always: function(callback) {
                     alwaysCB = callback;
                     return retObj;
                 }
                ,error: function(callback) {
                     errorCB = callback;
                     return retObj;
                 }
                ,run: function() {
                    // Construct the actual callback-function
                    var cbfun = function(error) {
                        if (error) {
                            // Log the error
                            module.logError(serverName, error);
                            // If supplied, run the error-callback
                            if (errorCB) errorCB.apply(this, arguments);
                            // If supplied, run the always-callback
                            if (alwaysCB) alwaysCB();
                            // Possibly run the quit-callback
                            if (quit) quitCB(error);
                        } else {
                            // If supplied, run the complete-callback
                            if (completeCB) completeCB.apply(this, Array.prototype.slice.call(arguments, 1));
                            // If supplied, run the always-callback
                            if (alwaysCB) alwaysCB();
                        }
                    };
                    // Call the original function with the supplied arguments, and append the callback-function
                    args.push(cbfun);
                    return func.apply(thisVal, args);
                 }
            }
            return retObj;
        };
    };

    /* WebSocket commands helper */
    module.expectCommand = function(connection, quit) {
        // Returns helper function that expects a message on the socket on the form of a JSON-object containing { "command": command, ... }
        // If it is received, it calls the callback function, if not, it calls the quit function (which usually closes the connection)
        return function(command, callback) {
            connection.once('message', function(msg) {
                // This should be a string
                if (msg.type == 'utf8') try {
                    // And should contain a JSON object
                    var data = JSON.parse(msg.utf8Data);
                    if (!data.command || data.command !== command) {
                        quit('Command "'+data.command+'" was not expected');
                    } else {
                        callback(data);
                    }
                } catch (error) {
                    quit(error);
                }
                else {
                    quit('Message was not a string');
                }
            });
        }
    };

    /* Create data-packets of output files */
    var tagIndRegx = /^([^-]*)-(\d+)/;
    module.packageSimulatorOutput = function(name, data) {
        // Numbers are all split over newlines
        var lines = data.split('\n');
        // Last line also contain \n, so last item will be empty
        lines.pop();

        // Convert all the strings back into numbers
        var nums = _.map(lines, function(line) { return parseFloat(line); });

        // Extragt tag and index from filename
        var m = name.match(tagIndRegx);
        var tag = m[1], index = m[2];
        var nameBuf = new Buffer(tag);

        /* Assemble package to send to client, we are sending:
            - Index of file (Uint16)
            - Number of numbers (Uint32)
            - Length of tag (Uint16)
            - Tag (utf8)
            - Numbers (Float32) */
        var buf = new Buffer(8+nameBuf.length+4*nums.length);
        buf.writeUInt16BE(index,0);
        buf.writeUInt32BE(nums.length,2);
        buf.writeUInt16BE(nameBuf.length,6);
        nameBuf.copy(buf,8);
        var numsStart = 8+nameBuf.length;
        for (var i = 0; i < nums.length; ++i) {
            buf.writeFloatBE(nums[i],numsStart+4*i);
        }

        return buf;
    };

    /* Kill a process and all of its child processes */
    module.killAll = function(process) {
        // Find all children of this process
        exec('ps -A -o pid,ppid', function(err, stdout, stderr) {
            if (!err) {
                var pids = {};
                _.each(stdout.split('\n'), function(line) {
                    var m = line.match(/^\s*(\d+)\s*(\d+)\s*$/);
                    if (m) {
                        pids[m[2]] = m[1];
                    }
                });
                // Make list of all children to kill
                var p = process.pid.toString();
                var children = [p];
                while (p = pids[p]) {
                    children.push(p);
                }
                // Try to kill all children
                spawn('kill', ['-9'].concat(children));
            } else {
                process.kill('SIGKILL');
            }
        });
    };

})(module.exports);
