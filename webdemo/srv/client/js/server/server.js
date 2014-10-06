(function(){
    /* Load dependencies for eqksServer module */
    angular.module('eqksServer', [
        'eqksConfiguration',
        // equelleToCpp.js
        'eqksIOStore',
        // cppToExecutable.js
        // executableRun.js
        'eqksFileHelpers'
    ]);
})();
