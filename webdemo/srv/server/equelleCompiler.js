/* Libraries */
var exec = require('child_process').exec,
    _ = require('underscore');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');

/* Compilation routine */
var compileEquelle = function(source, conn, quit, handleAnother) {
    var tryAsync = helpers.tryAsync('Equelle compiler', quit);

    /* Compile the executable using the Equelle compiler */
    var compiler = tryAsync(false, exec, config.equelle_compiler+' --input -')
    .complete(function(stdout) {
        // The compilation was successful, sign code with secret key
        tryAsync(helpers.signData, stdout)
        .complete(function(sign) {

            /* Parse inputs and outputs with compiler */
            var parser = tryAsync(false, exec, config.equelle_compiler+' --input - --dump io')
            .complete(function(iostdout) {
                // Parse dump-lines
                var inputs = [], outputs = [], current;
                _.each(iostdout.split('\n'), function(line) {
                    // Start definition of an input or output
                    if (line == 'Input') {
                        current = {};
                        inputs.push(current);
                    } else if (line == 'Output') {
                        current = {};
                        outputs.push(current);
                    } else {
                        // Description of input or output
                        var m;
                        if (m = line.match(/Tag: "((?:\\"|[^"])+)"/)) {
                            current.tag = m[1];
                        } else if (m = line.match(/Default: (.*)$/)) {
                            current.default = m[1];
                        } else if (m = line.match(/Type: (.*)$/)) {
                            current.type = m[1];
                        }
                    }
                });

                /* Done, send results to client */
                conn.sendJSON({ status: 'success', source: stdout, sign: sign, inputs: inputs, outputs: outputs});

                // Re-use socket to handle another compilation
                handleAnother();
            })
            .error(function(err, stdout, stderr) {
		        // Assuming that this is actual compilation errors
		        conn.sendJSON({ status: 'compileerror', err: stderr});
            })
            .run();

            parser.stdin.write(source);
            parser.stdin.end('\n');
        })
        .run();
    })
    .error(function(err, stdout, stderr) {
        // Assuming that this is actual compilation errors
        conn.sendJSON({ status: 'compileerror', err: stderr});
    })
    .run();

    /* Write the actual Equelle source to the compiler */
    compiler.stdin.write(source);
    // Add extra newline to parse last line of equelle code correctly
    compiler.stdin.end('\n');
}

/* The handleEquelleCompilerConnection(connection) function */
module.exports = function(handlerName, domain, conn, handleAnother) {
    var quit = function(error) { 
        // Send error to client
        conn.sendJSON({ status: 'failed', err: error.toString()});
        conn.close();
    };

    /* On receive compilation data from client */
    conn.once('message', function(mess) {
        if (mess.type != 'utf8') quit('Not utf8 data');
        else try {
            var data = JSON.parse(mess.utf8Data);
            if (!data.source) throw('No source received');

            // We should have all we need to compile the Equelle souce to C++
            compileEquelle(data.source, conn, quit, handleAnother);
        } catch (e) { quit(e) }
    });

    /* Let client know we are ready to receive data */
    conn.sendJSON({ status: 'ready' });

    /* Return the abortion function */
    // This thing runs so fast, that there is no need to abort it
    return function() {};
}
