/* Libraries */
var exec = require('child_process').exec;
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');

/* Compilation routine */
var compileEquelle = function(source, conn, errorCleanup) {
    /* Compile the executable using the Equelle compiler */
    var compiler = exec(config.equelle_compiler+' --input -', function(err, stdout, stderr) {
        /* Read compiled c++ code or errors */
        if (err) {
            /* Assuming that this is actual compilation errors */
            conn.sendJSON({ status: 'compilerror', err: stderr});
        } else {
            /* Sign the source code with our secret key */
            helpers.signData(stdout, function(err, sign) {
                if (err) errorCleanup(err);
                else {
                    conn.sendJSON({ status: 'success', source: stdout, sign: sign });
                    conn.close();
                }
            });
        }
    });
    compiler.stdin.write(source);
    compiler.stdin.end('\n'); // Add extra newline to parse last line of equelle code correctly
}

/* The handleEquelleCompilerConnection(connection) function */
module.exports = function(conn) {
    conn.sendJSON({ status: 'ready' });
    /* Error handling */
    var errorCleanup = function(err) {
        console.log((new Date())+': Error during Equelle compilation:');
        console.log(err);
        // Send error to client
        conn.sendJSON({ status: 'failed', err: err.toString()});
        conn.close();
    };
    /* On receive compilation data from client */
    conn.on('message', function(mess) {
        if (mess.type != 'utf8') errorCleanup('Not utf8 data');
        else try {
            var data = JSON.parse(mess.utf8Data);
            if (!data.source) throw('No source received');
            /* We should have all we need, start the executable compilation process */
            compileEquelle(data.source, conn, errorCleanup);
        } catch (e) { errorCleanup(e) }
    });
}
