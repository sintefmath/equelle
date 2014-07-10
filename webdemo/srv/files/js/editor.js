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
                /* Disable the user from going to the next step */
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
                ,gutters: ['equelle-gutter-error']
                ,matchBrackets: true
                ,autoCloseBrackets: { pairs: '()[]{}""||', explode: '{}' }
                ,tabSize: 4
                ,indentUnit: 4
            });
            /* Bind editor to compile-controller */
            controller.getSource = function() {
                return editor.getValue();
            };
            /* Show compilation errors in the editor */
            var errRegx = /^(Parser|Lexer).*line (\d+):\s*(.*)$/;
            var errorElement = $('<span class="glyphicon glyphicon-remove-sign equelle-gutter-error-sign text-danger"></span>')[0];
            controller.compiler.on('compileerror', function(err) {
                var firstLine = true;
                _.each(err.split('\n'), function(errmsg) {
                    var m = errmsg.match(errRegx);
                    if (m) {
                        var line = parseInt(m[2])-1;
                        var error = m[3];
                        var lineClass = 'equelle-line-error-'+line;
                        /* Set error mark in the gutter */
                        editor.setGutterMarker(line, 'equelle-gutter-error', errorElement);
                        editor.addLineClass(line, 'background', 'bg-danger');
                        editor.addLineClass(line, 'background', lineClass);
                        /* Find the line DOM element and add a tooltip with the actual error message */
                        var lineDOM = $(editor.getWrapperElement()).find('.'+lineClass).parent();
                        var preDOM, addTitle = function() {
                            preDOM = lineDOM.find('pre');
                            preDOM.attr('title',error);
                        };
                        addTitle();
                        /* Re add title every time the line is re-rendered */
                        var reAddTitle = function(event) {
                            if (event.target.nodeName == 'PRE') addTitle();
                        };
                        lineDOM.on('DOMNodeInserted', reAddTitle);
                        /* Remove classes and tooltip on recompile */
                        controller.compiler.once('started', function() {
                            editor.removeLineClass(line, 'background', 'bg-danger')
                            editor.removeLineClass(line, 'background', lineClass)
                            preDOM.removeAttr('title');
                            lineDOM.off('DOMNodeInserted', reAddTitle);
                        });
                        /* Scroll to the first error line */
                        if (firstLine) {
                            firstLine = false;
                            editor.scrollIntoView({line: line, ch: 0}, 100);
                        }
                    }
                });
            });
            /* Clear all error marks when compilation is rerun */
            controller.compiler.on('started', function() {
                editor.clearGutter('equelle-gutter-error');
            });
            /* Hook to documente change events */
            controller.compileTimer = null;
            editor.on('change', function() {
                controller.compiler.save(controller.getSource());
                scope.navigation.disableNext();
                $timeout.cancel(controller.compileTimer);
                controller.compileTimer = $timeout(function() {
                    controller.compiler.compile();
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
                controller.compiler.save(controller.getSource());
                controller.compiler.compile();
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
