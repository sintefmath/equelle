/* Libraries */
var _ = require('underscore'),
    spawn = require('child_process').spawn,
    exec = require('child_process').exec,
    fs = require('fs-extra');
/* Own modules */
var config = require('./config.js');

/* Helper function module */
(function(module) {
    ///* Reads all data from a stream into a string, then calls waiting function */
    //module.readAll = function(stream, endCB) {
    //    var data = '';
    //    stream.on('data', function(d) { data += d; });
    //    stream.on('end', function() { endCB(data); });
    //};

    ///* Reads all the data from stdout and stderr of given child process into strings, then calls waiting function */
    //module.readStdOutErr = function(process, doneCB) {
    //    var out = '', err = '';
    //    var done = _.after(2, function() {
    //        doneCB(out,err);
    //    });
    //    this.readAll(process.stdout, function(data) { out = data; done(); });
    //    this.readAll(process.stderr, function(data) { err = data; done(); });
    //};

    /* Generates a Message Authentication Code from buffer, then calls waiting function */
    module.signData = function(data, outCB) {
        /* Spawn OpenSSL and send the data */
        var signer = exec('openssl dgst -sha512 -hmac "'+config.secret_key+'"', function(err, stdout, stderr) {
            /* Retreive signature or errors */
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
        /* Spawn gzip which reads the file itself, and sends output to file.compressed */
        var compress = exec('gzip -c -9 '+file+' > '+file+'.compressed', function(err, stdout, stderr) {
            if (err) outCB(stderr);
            else {
                /* Read the compressed file into a buffer, and call waiting function */
                fs.readFile(file+'.compressed', function(err, compressed) {
                    if (err) outCB(err);
                    else outCB(null, compressed);
                });
            }
        });
    };

    /* Decompresses a buffer into a file, calls waiting function when done */
    module.decompressToFile = function(file, data, outCB) {
        /* Spawn gzip which writes decompressed data to file, and send buffer to stdin */
        var decompress = exec('gzip -d - > '+file, function(err, stdout, stderr) {
            /* Read potential errors */
            if (err) outCB(stderr);
            else outCB(null);
        });
        decompress.stdin.end(data);
    };
})(module.exports);
