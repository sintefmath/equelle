/* Own modules */
var handleEquelleCompileConnection = require('./equelleCompiler.js'),
    handleExecutableCompileConnection = require('./executableCompiler.js'),
    handleExecutableRunConnection = require('./executableRun.js');

(function(module) {
    /* Function which checks the origin of the connecting socket to determine if it should be allowed */
    module.shouldAccept = function(origin) {
        // TODO: Implement this!
        return true;
    };

    /* Function which returns the appropriate handler function for a given protocol */
    module.acceptHandler = function(protocol) {
        switch (protocol) {
            // Compile from Equelle to C++
            case 'equelle-compile':
                return handleEquelleCompileConnection;
            // Compile an executable from C++
            case 'executable-compile':
                return handleExecutableCompileConnection;
            // Run executable simulator
            case 'executable-run':
                return handleExecutableRunConnection;
            default:
                return undefined;
        }
    };
})(module.exports);
