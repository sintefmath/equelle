(function(){
    angular.module('eqksConfiguration', [])
    .factory('eqksConfig', function() {
        /* This module contains all runtime variables used throughout the application*/
        var localStorageTagPrefix = 'eqks';
        var lsPrefixTag = function(tag) { return localStorageTagPrefix+tag };
        // TODO: Read all localStorage tags from this class!
        // TODO: Document what all these settings do!
        return {
             compileHost: window.location.host
            ,runHost: window.location.host
            ,paramFileName: 'eqksparams.param'
            ,executableName: 'simulator'
            ,localStorageTags: {
                 prefix: localStorageTagPrefix
                ,inputs: lsPrefixTag('Inputs')
                ,inputFile: lsPrefixTag('InputsFile-')
                ,equelleSource: lsPrefixTag('Source') 
                ,cppSource: lsPrefixTag('Compiled') 
                ,cppSourceSignature: lsPrefixTag('CompiledSign') 
                ,grid: lsPrefixTag('Grid')
                ,executable: lsPrefixTag('Executable')
                ,executableSignature: lsPrefixTag('ExecutableSign')
                ,executableSourceSignature: lsPrefixTag('ExecutableSourceSign')
                ,previousThumbnail: lsPrefixTag('Thumbnail')
            }
            ,grid: {
                 maxSize: 300
                ,defaults: {
                     dimensions: 2
                    ,size: [10, 10, 10]
                    ,cellSize: [1, 1, 1]
                }
            }
    }})
})();
