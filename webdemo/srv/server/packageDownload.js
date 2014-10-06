/* Libraries */
var tmp = require('tmp'),
    fs = require('fs-extra');
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js');

(function(module) {
    /* This function returns the packaged output from a simulator run to the client */
    module.getPackage = function(name, res) {
        var quit = function(error) { res.writeHead(404); res.end() };
        var tryAsync = helpers.tryAsync('Output package server', quit);

        var path = tmp.tmpdir+'/equelleOutputPackage'+name;
        // Read the file if it exists, if not, auto-respond with error
        tryAsync(true, fs.readFile, path)
        .complete(function(data) {
            // Now delete the temporary file, we don't care if it works or not
            tryAsync(fs.remove, path).run();

            // Write the data of the package to response
            res.writeHead(200, {
                'Content-Type': 'application/x-gtar',
                'Content-Length': data.length,
                'Content-Disposition': 'attachment; filename="outputs.tar.gz"'
            });
            res.end(data);
        })
        .run();
    };
})(module.exports);
