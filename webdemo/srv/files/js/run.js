(function(){
    /* This module provides the routines for setting input variables and uploading files */
    angular.module('equelleKitchenSinkRun', ['equelleKitchenSinkExecutableRun'])
    /* Connect to the server socket */
    .directive('eqksRun', ['$timeout','localStorageFile','executableRun', function($timeout, localStorageFile, executableRun) { return {
         restrict: 'A'
        ,controller: function($scope) {
            $scope.working = false;
            $scope.progress = 0;
            $scope.output = '';
        }
        ,link: function(scope, elem, attrs, controller) {
            var executer = new executableRun.Executer();
            /* Hook to events */
            var stdout = '', stderr = '';
            executer.on('started', function() {
                scope.working = true;
            });
            executer.on('progress', function(p) {
                scope.$apply(function() { scope.progress = p });
                scope.progress = p;
            });
            executer.on('complete', function() {
                scope.$apply(function() {
                    scope.progress = 100;
                    working = false;
                });
            });

            executer.on('stdout', function(data) {
                stdout += data;
                scope.$apply(function() { scope.output = stdout });
            });
            executer.on('stderr', function(data) { stderr += data });
            /* Start the simlation */
            executer.run();
        }
    }}])
})();
