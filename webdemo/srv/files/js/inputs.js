(function(){
    /* This module provides the routines for setting input variables and uploading files */
    angular.module('equelleKitchenSinkInputs', ['equelleKitchenSinkHelpers', 'equelleKitchenSinkEquelleCompile', 'equelleKitchenSinkEquelleInputs', 'equelleKitchenSinkExecutableCompile'])
    /* The source-scanning routine */
    .directive('eqksInputs', ['$timeout', 'localStorageFile', 'equelleCompiler', 'equelleInputs', 'executableCompiler',
        function($timeout, localStorageFile, equelleCompiler, equelleInputs, executableCompiler) { return {
         restrict: 'A'
        ,controller: function($scope) {
            /* Check the Equelle source has been compiled correctly */
            if (!equelleCompiler.hasCompiled()) {
                $scope.error = 'No Equelle code has been compiled, please <a href="#/editor/">go back</a> and provide code.';
            } else {
                /* Initial parsing */
                equelleInputs.parse();
                /* Get the inputs for use in children */
                $scope.singleScalars = equelleInputs.getSingleScalars();
                $scope.files = equelleInputs.getFiles();
            }
            /* Configuration for the dropzones */
            this.dropzoneConfig = {
                 url: 'upload.php' // Dummy, we don't actually want to send the files anywhere
                ,accept: function() { return false } // Stop the Dropzone.js from uploading files
                ,createImageThumbnails: false // We don't want to do this
                ,clickable: true
                ,uploadMultiple: false // TODO: Why doesn't this work?
            };
            /* Update inputs if values are changed */
            $scope.$on('inputsChanged', function() {
                $scope.$apply($scope.singleScalars = equelleInputs.getSingleScalars());
                $scope.$apply($scope.files = equelleInputs.getFiles());
            });
        }
        ,link: function(scope, elem, attrs, controller) {
            /* Start compiling the executable file if the result of the C++ source code is not the same as the one we have */
            if (executableCompiler.hasCompiled()) {
                scope.navigation.enableNext();
            } else {
                /* Wait a little, just in case the user jumps back */
                var execCompileWait = $timeout(function() {
                    var compiler = new executableCompiler.Compiler();
                    /* Compilation started */
                    compiler.on('started', function() {
                        scope.navigation.laddaNext.start();
                    });
                    /* Show compilation progress */
                    compiler.on('progress', function(prog) {
                        /* Receives 0-100, Ladda button expects 0-1 */
                        scope.navigation.laddaNext.setProgress(prog/100);
                    });
                    /* Compilation failed from an internal error */
                    compiler.on('failed', function(err) {
                        console.log('Internal executable compilation error:');
                        console.log(err);
                        // TODO: Log these errors somewhere?
                        scope.navigation.laddaNext.setProgress(0);
                        scope.navigation.laddaNext.stop();
                        scope.navigation.disableNext();
                    });
                    /* Compilation was successful */
                    compiler.on('completed', function() {
                        scope.navigation.laddaNext.setProgress(0);
                        scope.navigation.laddaNext.stop();
                        scope.navigation.enableNext();
                    });
                    compiler.compile();
                }, 4000);
                // TODO: Abort compilation if user navigates away from page?
                elem.on('$destroy', function() {
                    $timeout.cancel(execCompileWait);
                });
            }
        }
 
    }}])
    /* The file upload dropzone */
    .directive('eqksInputsDropzone', ['eqksConfig','equelleInputs','localStorageFile', function(eqksConfig, equelleInputs,localStorageFile) { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            /* Create a new file Dropzone for this input */
            var dropzone = new Dropzone(elem.context, controller.dropzoneConfig);
            /* For some reason, the click events does not bubble up to the dropzone div element, so we bind it ourself */
            var el = $(elem.context);
            el.find('b').click(function() { el.click() });
            /* Handle file uploads */
            dropzone.on('addedfile', function(file) {
                /* Stop the Dropzone.js from doing anything else */
                this.removeFile(file);
                /* Now we can save the file to localStorage ourself */
                var name = eqksConfig.localStorageTags.inputFile+scope.input.tag;
                localStorageFile.write(name, file, function(err) {
                    /* Let the inputs-class know we have got the file */
                    if (err) equelleInputs.setValue(scope.input.tag,undefined);
                    else equelleInputs.setValue(scope.input.tag,file.name);
                    scope.$emit('inputsChanged');
                });
            });
        }
    }}])
    /* Single scalar value inputs */
    .directive('eqksInputsSingleScalar', ['eqksConfig','equelleInputs', function(eqksConfig, equelleInputs) { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            elem.on('change', function() {
                equelleInputs.setValue(scope.input.tag, !!this.value ? this.value : undefined);
                scope.$emit('inputsChanged');
            });
        }
    }}])
    /* Simulation grid specification */
    .directive('eqksInputsGrid', ['eqksConfig','equelleInputs', function(eqksConfig, equelleInputs) { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,controller: function($scope) {
            /* Get and set grid options from localStorage here */
            $scope.maxSize = eqksConfig.grid.maxSize;
            $scope.defaults = eqksConfig.grid.defaults;
            $scope.grid = equelleInputs.getGrid();
            $scope.$watch('grid', function() {
                /* The grid-setting values has changed, save to localStorage */
                equelleInputs.setGrid($scope.grid);
            }, true);
        }
    }}])
    /* Positive integers input element */
    .directive('eqksInputsPositiveInteger', function() { return {
         restrict: 'A'
        ,link: function(scope, elem, attrs) {
            var max = parseFloat(attrs.max);
            if (isNaN(max)) max = Infinity;
            // TODO: Need support for more browsers and OSX here?
            var prevValue;
            elem.on('keypress', function(event) {
                prevValue = this.value;
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers in this box */
                else if (!/[0-9]/.test(ch)) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum */
                if (parseInt(this.value) > max) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                var i = parseInt(text);
                /* Only allow pasting if it is a positive integer */
                if (isNaN(i) || i < 1 || i > max) event.preventDefault();
            });
        }
    }})
    /* Float input element */
    .directive('eqksInputsFloat', function() { return {
         restrict: 'A'
        ,link: function(scope, elem, attrs) {
            var max = parseFloat(attrs.max), min = parseFloat(attrs.min);
            if (isNaN(max)) max = Infinity;
            if (isNaN(min)) min = -Infinity;
            // TODO: Need support for more browsers and OSX here?
            var prevValue;
            elem.on('keypress', function(event) {
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers in this box */
                else if (!/[\-0-9.]/.test(ch)) event.preventDefault();
                /* Only allow one . */
                else if (ch == '.' && _.str.contains(this.value,'.')) event.preventDefault();
                /* Only allow one - at the very beginning */
                else if (ch == '-' && this.selectionStart != 0) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum and under minimum */
                var f = parseFloat(this.value);
                if (f > max || f < min) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                /* Only allow pasting if it is a float */
                if (isNaN(parseFloat(text))) event.preventDefault();
            });
        }
    }})
    /* Single scalar value inputs */
    .directive('eqksInputsPositiveFloat', function() { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            var max = parseFloat(attrs.max);
            if (isNaN(max)) max = Infinity;
            // TODO: Need support for more browsers and OSX here?
            var prevValue;
            elem.on('keypress', function(event) {
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers and . */
                else if (!/[0-9.]/.test(ch)) event.preventDefault();
                /* Only allow one . */
                else if (ch == '.' && _.str.contains(this.value,'.')) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum */
                if (parseFloat(this.value) > max) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                var f = parseFloat(text);
                /* Only allow pasting if it is a positive float */
                if (isNaN(f) || f <= 0) event.preventDefault();
            });
        }
    }})
})();
