/* Includes */
var http = require('http'),
    spawn = require('child_process').spawn,
    _ = require('underscore');

/* Config */
var equelle_dir = '/equelle/build/compiler',
    equelle_compiler = equelle_dir+'/ec';

/* Helpers */
var readAll = function(stream, endCB) {
    var data = '';
    stream.on('data', function(d) { data += d; });
    stream.on('end', function() { endCB(data); });
}


/* Server logic */
var server = http.createServer(function(req,res) {
    if (req.method != 'POST') {
        // Not allowed request
        res.statusCode = 403;
        res.end();
    } else {
        // Read source code from POST data
        readAll(req, function(source) {
            // Try to compile the source
            compiler = spawn(equelle_compiler, ['--input','-']);
            compiler.stdin.write(source);
            compiler.stdin.end('\n'); // Add extra newline to parse last line of equelle code correctly
            // Read compiled cpp code or errors
            var out = '', err = '';
            var done = _.after(2, function() {
                var ret = { out: out, err: err };
                res.end(JSON.stringify(ret));
            });
            readAll(compiler.stdout, function(data) { out = data; done(); });
            readAll(compiler.stderr, function(data) { err = data; done(); });
        });
    }
}).listen(8880);
console.log('Server started');
