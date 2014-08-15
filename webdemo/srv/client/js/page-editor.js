(function(){
    angular.module('eqksPageEditor', [ 'eqksUi',
                                       'eqksServer'
    ])
    /* The Editor-page controller */
    .controller('editorController', ['$scope','$timeout','eqksEquelleToCpp', function($scope, $timeout, compiler) {
        /* Header and navigation */
        $scope.title = 'Edit code:';
        $scope.navigation = {
             previous: { path: '/', text: 'Select project' }
            ,next: { path: '/inputs/', text: 'Provide inputs', disabled: !compiler.hasCompiled() }
        };

        /* CodeMirror editor */
        $scope.editor = {
            source: compiler.source(),
            errors: []
        };

        /* Non-compilation errors */
        $scope.error = {
            show: false,
            text: ''
        };

        /* Save changes of the source code, and auto-compile after a break */
        var compileTimer;
        $scope.$watch('editor.source', function() {
            compiler.save($scope.editor.source);
            $timeout.cancel(compileTimer);
            $timeout(function() {
                compiler.compile();
            }, 2000);
        });

        /* Bind to compiler events */
        var compilerStarted = function() {
            $scope.compileButton.working = true;
        };
        var compilerCompleted = function() {
            $scope.$apply(function() {
                // Buttons
                $scope.compileButton.working = false;
                $scope.compileButton.class = 'success';
                $scope.navigation.next.disabled = false;
                // Compiler errors
                $scope.error.show = false;
                // Compilation errors
                $scope.editor.errors = [];
            });
        };
        var compilerError = function(errors) {
            $scope.$apply(function() {
                // Buttons
                $scope.compileButton.working = false;
                $scope.compileButton.class = 'danger';
                $scope.navigation.next.disabled = true;
                // Compiler errors
                $scope.error.show = false;
                // Compilation errors
                $scope.editor.errors = errors;
            });
        };
        var compilerFailed = function(error) {
            $scope.$apply(function() {
                // Buttons
                $scope.compileButton.working = false;
                $scope.compileButton.class = 'danger';
                $scope.navigation.next.disabled = true;
                // Compiler errors
                $scope.error.text = '<strong>Compiler error:</strong> '+error;
                $scope.error.show = true;
                // Compilation errors
                $scope.editor.errors = [];
            });
        };
        compiler.on('started', compilerStarted);
        compiler.on('completed', compilerCompleted);
        compiler.on('compileerror', compilerError);
        compiler.on('failed', compilerFailed);
        // Remove bindings once scope is gone
        $scope.$on('$destroy', function() {
            compiler.off('started', compilerStarted);
            compiler.off('completed', compilerCompleted);
            compiler.off('compileerror', compilerError);
            compiler.off('failed', compilerFailed);
        });

        /* Compilation button */
        $scope.compileButton = {
            working: false,
            class: 'primary'
        };
        $scope.onCompileClick = function(event) {
            compiler.compile();
        };
    }])
})();
