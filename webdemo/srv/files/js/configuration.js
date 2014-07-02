(function(){
    angular.module('equelleKitchenSinkConfiguration', [])
    .factory('eqksConfig', function() {
        /* This module contains all runtime variables used throughout the application*/
        return {
            compileHost: window.location.host
    }})
})();
