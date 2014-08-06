/* Libraries */
var exec = require('child_process').exec;
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');

/* Compilation routine */
var compileEquelle = function(source, conn, quit) {
    var tryAsync = helpers.tryAsync('Equelle compiler', quit);

    /* Compile the executable using the Equelle compiler */
    var compiler = tryAsync(false, exec, config.equelle_compiler+' --input -')
    .complete(function(stdout, stderr) {
        // The compilation was successful, sign code with secret key
        tryAsync(helpers.signData, stdout)
        .complete(function(sign) {
            conn.sendJSON({ status: 'success', source: stdout, sign: sign });
            conn.close();
        })
        .run();
    })
    .error(function(err, stdout, stderr) {
        // Assuming that this is actual compilation errors
        conn.sendJSON({ status: 'compilerror', err: stderr});
    })
    .run();

    /* Write the actual Equelle source to the compiler */
    compiler.stdin.write(source);
    // Add extra newline to parse last line of equelle code correctly
    compiler.stdin.end('\n');
}

/* The handleEquelleCompilerConnection(connection) function */
module.exports = function(conn) {
    var quit = function(error) { 
        // Send error to client
        conn.sendJSON({ status: 'failed', err: err.toString()});
        conn.close();
    };

    /* On receive compilation data from client */
    conn.on('message', function(mess) {
        if (mess.type != 'utf8') quit('Not utf8 data');
        else try {
            var data = JSON.parse(mess.utf8Data);
            if (!data.source) throw('No source received');

            // We should have all we need to compile the Equelle souce to C++
            compileEquelle(data.source, conn, quit);
        } catch (e) { quit(e) }
    });

    /* Let client know we are ready to receive data */
    conn.sendJSON({ status: 'ready' });
}
