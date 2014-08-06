/* Libraries */
var http = require('http'),
    websocket = require('websocket').server;
/* Own modules */
var config = require('./config.js'),
    helpers = require('./helpers.js'),
    handleEquelleCompileConnection = require('./equelleCompiler.js'),
    handleExecutableCompileConnection = require('./executableCompiler.js'),
    handleExecutableRunConnection = require('./executableRun.js');
var examples = require('./examples.js');


/* Setup the HTTP server */
var httpServer = http.createServer(function(req,res) {
    // Respond to all non-websocket requests with a 404
    res.writeHead(404);
    res.end();
}).listen(8888, function() {
    console.log((new Date())+': Socket HTTP server was started');
});

/* Setup the WebSocket server */
var socketServer = new websocket({
    httpServer: httpServer,
    autoAcceptConnections: false,
    maxReceivedFrameSize: 10*1024*1024, //10MiB
    maxReceivedMessageSize: 10*1024*1024 //10MiB
});

/* Handle a connection from a client */
socketServer.on('request', function(req) {
    if (false) {
        //TODO: Check if origin of the request is allowed!!
        req.reject();
    } else {
        // Accept connection to client, and add JSON sending method for convenience
        var acceptConnection = function(request, protocol) {
            var connection = request.accept(protocol, request.origin);
            connection.sendJSON = function(obj) { connection.sendUTF(JSON.stringify(obj)) };
            return connection;
        };
        // Connection to this client is allowed, where does it want to connect?
        if (req.requestedProtocols.length != 1) req.reject();
        else switch (req.requestedProtocols[0]) {
            // Compile from Equelle to C++
            case 'equelle-compile':
            handleEquelleCompileConnection(acceptConnection(req,'equelle-compile'));
            break;
            // Compile an executable from C++
            case 'executable-compile':
            handleExecutableCompileConnection(acceptConnection(req,'executable-compile'));
            break;
            // Run executable simulator
            case 'executable-run':
            handleExecutableRunConnection(acceptConnection(req,'executable-run'));
            break;
            // Not a recognised protocol, drop connection
            default:
            req.reject();
        }
    }
});


/* Examples listing server */
var exServer = http.createServer(function(req,res) {
    var i;
    if (req.url == '/list.json') {
        // Respond with a list of all available examples
        examples.getList(res);
    } else if ((i = req.url.indexOf('/',1)) > 0) {
        // Client is trying to get an example input-file
        examples.getInputFile(req.url.substr(1,i-1), req.url.substr(i+1), res);
    } else {
        // Client is trying to get the data for an example
        examples.getExample(req.url.substr(1), res);
    }
}).listen(8887, function() {
    console.log((new Date())+': Examples HTTP server was started');
});
