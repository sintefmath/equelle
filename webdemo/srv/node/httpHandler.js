/* Own modules */
var examples = require('./examples.js'),
    packageDownload = require('./packageDownload.js');

/* httpHandler module */
(function(module) {
    /* The function that handles normal requests to the HTTP-server */
    module.handleRequest = function(handlerName, domain, req, res) {
        var i;
        if (req.url == '/list.json') {
            // Respond with a list of all available examples
            examples.getList(res);
        } else if (req.url.indexOf('/outputpackage/') == 0) {
            // Client is trying to download a package of the output files
            packageDownload.getPackage(req.url.substr(15), res);
        } else if ((i = req.url.indexOf('/',1)) > 0) {
            // Client is trying to get an example input-file
            examples.getInputFile(req.url.substr(1,i-1), req.url.substr(i+1), res);
        } else {
            // Client is trying to get the data for an example
            examples.getExample(req.url.substr(1), res);
        }
    };
})(module.exports);
