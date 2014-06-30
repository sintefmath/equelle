(function(){
    /* This module provides the editor and Equelle->C++ compilation routines */
    angular.module('equelleKitchenSinkEditor', [])
    /* The compilation routine */
    .directive('eqksEquelleCompiler', function() { return {
         restrict: 'A'
        ,controller: function($scope) {
            this.compile = function() {
                localStorage.source = this.getSource();
                localStorage.compiled = '';
                localStorage.compiledSign = '';
                this.setButtonWorking();
                controller = this;
                $.post('/equellecompiler/', localStorage.source, null, 'json').done(function(data) {
                    console.log(data);
                    if (data.err) {
                        // Compiler error
                        controller.showError('<b>Compiler error:</b><br><pre>'+data.err+'</pre');
                        controller.setButtonError();
                        $scope.navigation.disableNext();
                    } else if (data.out) {
                        // Completed, save result
                        localStorage.compiled = data.out;
                        localStorage.compiledSign = data.sign;
                        controller.hideError();
                        controller.setButtonComplete();
                        $scope.navigation.enableNext();
                    } else {
                        controller.showError('An internal error occured during compilation to C++');
                        controller.setButtonError();
                        $scope.navigation.disableNext();
                        // TODO: Log error
                    }
                }).fail(function() {
                    controller.showError('An internal error occured during compilation to C++');
                    controller.setButtonError();
                    $scope.navigation.disableNext();
                    // TODO: Log error
                });
            }
        }
        ,link: function(scope, elem, attrs, controller) {
            /* Check if the code has already been compiled */
            if (localStorage.compiled && localStorage.compiledSign) {
                controller.hideError(),
                controller.setButtonComplete();
                scope.navigation.enableNext();
            }
        }
    }})
    /* Editor window */
    .directive('eqksEquelleEditor', ['$timeout', function($timeout) { return {
         restrict: 'A'
        ,require: '^eqksEquelleCompiler'
        ,link: function(scope, elem, attrs, controller) {
            var editor = ace.edit(elem.context);
            editor.setValue(localStorage.source);
            editor.gotoLine(0,0);
            editor.session.setNewLineMode('unix');
            // Bind editor to compile-controller
            controller.getSource = function() {
                return editor.getValue();
            };
            // Hook to documente change events
            var compileTimer;
            editor.on('change', function() {
                $timeout.cancel(compileTimer);
                compileTimer = $timeout(function() {
                    controller.compile();
                },3000);
            });
            // Cleanup when template is deleted from DOM
            elem.on('$destroy', function() { editor.destroy() });
        }
    }}])
    /* Compile button */
    .directive('eqksEquelleCompilerButton', function() { return {
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
                controller.compile();
            });
        }
    }})
    /* Error window */
    .directive('eqksEquelleCompilerError', function() { return {
         restrict: 'A'
        ,require: '^eqksEquelleCompiler'
        ,link: function(scope, elem, attrs, controller) {
            var el = $(elem.context);
            // Bind window to compile-controller
            controller.showError = function(text) { el.html(text).removeClass('hide') };
            controller.hideError = function() { el.addClass('hide') };
            // Initially hide window
            el.addClass('hide');
        }
    }})
})();
