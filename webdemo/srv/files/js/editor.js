(function(){
    /* This module provides the editor and Equelle->C++ compilation routines */
    angular.module('equelleKitchenSinkEditor', ['equelleKitchenSinkEquelleCompile','equelleKitchenSinkConfiguration'])
    /* The compilation routine */
    .directive('eqksEquelleCompiler', ['equelleCompiler', function(equelleCompiler) { return {
         restrict: 'A'
        ,controller: function($scope) {
            var controller = this;
            var compiler = controller.compiler = new equelleCompiler.Compiler();
            /* Compilation started */
            compiler.on('started', function() {
                controller.setButtonWorking();
            });
            /* Compilation failed from an internal error */
            compiler.on('failed', function(err) {
                console.log('Internal equelle compilation error:');
                console.log(err);
                // TODO: Log these errors somewhere?
                controller.setButtonError();
                $scope.navigation.disableNext();
            });
            /* Compilation failed due to a source-code error */
            compiler.on('compileerror', function(err) {
                controller.showError(err);
                controller.setButtonError();
                $scope.navigation.disableNext();
            });
            /* Compilation was successful */
            compiler.on('completed', function() {
                controller.hideError();
                controller.setButtonComplete();
                $scope.navigation.enableNext();
            });
        }
        ,link: function(scope, elem, attrs, controller) {
            /* Check if the code has already been compiled */
            if (equelleCompiler.hasCompiled()) {
                controller.hideError(),
                controller.setButtonComplete();
                scope.navigation.enableNext();
            }
        }
    }}])
    /* Editor window */
    .directive('eqksEquelleEditor', ['$timeout','eqksConfig', function($timeout, eqksConfig) { return {
         restrict: 'A'
        ,require: '^eqksEquelleCompiler'
        ,link: function(scope, elem, attrs, controller) {
            /* Create an editor */
            var editor = CodeMirror(elem.context, {
                 value: localStorage.getItem(eqksConfig.localStorageTags.equelleSource)
                ,mode: 'equelle'
                ,lineNumbers: true
            });
            /* Bind editor to compile-controller */
            controller.getSource = function() {
                return editor.getValue();
            };
            /* Hook to documente change events */
            controller.compileTimer = null;
            editor.on('change', function() {
                scope.navigation.disableNext();
                $timeout.cancel(controller.compileTimer);
                controller.compileTimer = $timeout(function() {
                    controller.compiler.compile(controller.getSource());
                },3000);
            });
            /* Cleanup when template is deleted from DOM */
            elem.on('$destroy', function() { $timeout.cancel(controller.compileTimer) });
        }
    }}])
    /* Compile button */
    .directive('eqksEquelleCompilerButton', ['$timeout', function($timeout) { return {
         restrict: 'A'
        ,require: '^eqksEquelleCompiler'
        ,link: function(scope, elem, attrs, controller) {
            var el = $(elem.context).addClass('ladda-button');
            var spinner = Ladda.create(elem.context);
            // Bind button to compile-controller
            controller.setButtonWorking = function() { spinner.start() };
            controller.setButtonComplete = function() { el.removeClass('btn-primary btn-danger').addClass('btn-success'); spinner.stop() };
            controller.setButtonError = function() { el.removeClass('btn-primary btn-success').addClass('btn-danger'); spinner.stop() };
            // Hook to click event
            el.click(function() {
                $timeout.cancel(controller.compileTimer);
                controller.compiler.compile(controller.getSource());
            });
        }
    }}])
    /* Error window */
    .directive('eqksEquelleCompilerError', function() { return {
         restrict: 'A'
        ,require: '^eqksEquelleCompiler'
        ,link: function(scope, elem, attrs, controller) {
            var el = $(elem.context);
            // Bind window to compile-controller
            controller.showError = function(text) { el.html(text).removeClass('hide') };
            controller.hideError = function() { el.addClass('hide').html('') };
            // Initially hide window
            el.addClass('hide');
        }
    }})
})();
