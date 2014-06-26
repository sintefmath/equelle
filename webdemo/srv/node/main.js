/* Includes */
var http = require('http'),
    spawn = require('child_process').spawn,
    exec = require('child_process').exec,
    fs = require('fs'),
    tmp = require('tmp'),
    _ = require('underscore');

/* Config */
var equelle_dir = '/equelle/build',
    equelle_compiler = equelle_dir+'/compiler/ec';
var secret_key = 'equellesecret'; /* TODO: how to keep a key secret for use in production? */
var compiler_skel_dir = '/scripts/cppcompilerskel';

/* Helpers */
var readAll = function(stream, endCB) {
    var data = '';
    stream.on('data', function(d) { data += d; });
    stream.on('end', function() { endCB(data); });
};

var readStdOutErr = function(process, doneCB) {
    var out = '', err = '';
    var done = _.after(2, function() {
        doneCB(out,err);
    });
    readAll(process.stdout, function(data) { out = data; done(); });
    readAll(process.stderr, function(data) { err = data; done(); });
};

var signData = function(data, outCB) {
    // Send data to OpenSSL
    var signer = spawn('openssl', ['dgst','-sha512','-hmac','"'+secret_key+'"']);
    signer.stdin.end(data);
    // Retreive signature
    readAll(signer.stdout, function(out) {
        outCB(out.substring(9,out.length-1));
    });
};

/* Equelle compiler server */
var equelleSrv = http.createServer(function(req,res) {
    if (req.method != 'POST') {
        // Not allowed request
        res.statusCode = 403;
        res.end();
    } else {
        // Read source code from POST data
        readAll(req, function(source) {
            console.log('Got compiler request');
            // Try to compile the source
            var compiler = spawn(equelle_compiler, ['--input','-']);
            compiler.stdin.write(source);
            compiler.stdin.end('\n'); // Add extra newline to parse last line of equelle code correctly
            // Read compiled cpp code or errors
            readStdOutErr(compiler, function(out, err) {
                console.log('Compiler out:');
                console.log(out);
                console.log('Compiler err:');
                console.log(err);
                var ret = { out: out, err: err };
                if (!ret.err) {
                    signData(out, function(sign) {
                        ret.sign = sign;
                        res.end(JSON.stringify(ret));
                        console.log('Send compiler result:');
                        console.log(ret);
                    });
                } else {
                    res.end(JSON.stringify(ret));
                }
            });
        });
    }
}).listen(8880);

/* CPP compiler server */
var cppSrv = http.createServer(function(req,res) {
    // TODO: Lots of signature verification and error-checking to do here!
    if (req.method != 'POST') {
        // Not allowed request
        res.statusCode = 403;
        res.end();
    } else {
        readAll(req, function(data) {
            // Make a temporary folder for this compilation
            tmp.dir({prefix: 'equelletmp'}, function(err, dir) {
                console.log('Working in tmp-dir: '+dir);
                // Copy make-skeleton
                var cp = exec('cp -R '+compiler_skel_dir+'/* '+dir);
                cp.on('exit', function(code) {
                    console.log('Copied files with return code: '+code);
                    if (code == 0) {
                        // Write cpp source to folder
                        fs.writeFile(dir+'/simulator.cpp', data, function(err) {
                            var cmake = spawn('cmake', ['-DEquelle_DIR='+equelle_dir], {cwd: dir});
                            readStdOutErr(cmake, function(out, err) {
                                console.log('CMake out:');
                                console.log(out);
                                console.log('CMake err:');
                                console.log(err);
                                if (!err) {
                                    var make = spawn('make', [], {cwd: dir});
                                    readStdOutErr(make, function(out, err) {
                                        console.log('Make out:');
                                        console.log(out);
                                        console.log('CMake err:');
                                        console.log(err);
                                        // Sign, compress and send executable
                                        fs.readFile(dir+'/simulator', function(err, executable) {
                                            var output = {};
                                            var done = _.after(2, function() {
                                                if (!output.err) {
                                                    res.write('success:');
                                                    res.write(output.sign);
                                                    res.write(':');
                                                    res.write(output.compressed);
                                                }
                                                res.end();
                                            });
                                            signData(executable, function(sign) {
                                                output.sign = sign;
                                                done();
                                            });
                                            var compress = spawn('gzip', ['-c', dir+'/simulator']);
                                            readStdOutErr(compress, function(out, err) {
                                                if (!err) {
                                                    output.compressed = out;
                                                    done();
                                                }
                                            });
                                        });
                                        var sign = sp
                                    });
                                }
                            });
                        });
                    } else fs.rmdir(dir);
                });
            });

            //console.log('Got cpp compiler request:');
            //console.log(data);
        });
    }
}).listen(8881);

/* Simulator run server */
var runSrv = http.createServer(function(req,res) {
    if (req.method != 'POST') {
        // Not allowed request
        res.statusCode = 403;
        res.end();
    } else {
        // This server receives a stream of file on the form [len_filename|filename, len_data|data]+
        // Read all these files, and put them in a temporary directory
        // TODO: Lots of error handling
        tmp.dir({prefix: 'equelle_run_'}, function(err, dir) {
        });
    }
}).listen(8882);

console.log('Server started');
