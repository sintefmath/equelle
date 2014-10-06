(function(){
    angular.module('eqksPageInputs', [ 'eqksUi',
                                       'eqksServer',
                                       'eqksIOStore',
                                       'eqksConfiguration'
    ])
    /* The Inputs-page controller */
    .controller('inputsController', ['$scope','$timeout','eqksInputs','eqksGrid','eqksCppToExecutable', function($scope, $timeout, inputs, grid, compiler) {
        /* Header and navigation */
        $scope.title = 'Provide inputs:';
        $scope.navigation = {
             previous: { path: '/editor/', text: 'Edit code' }
            ,next: { path: '/run/', text: 'Run simulation', disabled: !compiler.hasCompiled() }
        };

        /* Inputs for this simulator */
        $scope.inputs = {
            singleScalars: inputs.getSingleScalars(),
            files: inputs.getFiles()
        };
        // Update inputs if values are changed
        $scope.$on('inputsChanged', function() {
            $timeout(function() {
                $scope.inputs.singleScalars = inputs.getSingleScalars();
                $scope.inputs.files = inputs.getFiles();
            });
        });
        // Write values when changed in inputs
        $scope.setInput = function(tag, value) {
            inputs.setValue(tag, value);
            $scope.$emit('inputsChanged');
        };

        /* Grid for this simulator */
        $scope.grid = grid.get();
        $scope.$watch('grid', function() {
            $scope.grid = grid.set($scope.grid);
        }, true);

        /* Compiler button */
        $scope.compileButton = {
            working: false,
            class: 'primary'
        };
        $scope.onCompileClick = function(event) {
            compiler.compile();
        };
        compiler.bindButtonProperties($scope.compileButton, $scope);

        /* Update next navigation button on compilation progress */
        var updateNext = function() { $scope.$apply(function() { $scope.navigation.next.disabled = !compiler.hasCompiled() }) };
        compiler.on('completed', updateNext );
        compiler.on('failed', updateNext );
    }])
    /* The file upload dropzone */
    .directive('eqksInputsDropzone', ['eqksConfig','eqksInputs','localStorageFile', function(config,inputs,lsFile) { return {
         restrict: 'A'
        ,link: function($scope, elem, attrs, controller) {
            /* Create a new file Dropzone for this input */
            var dropzone = new Dropzone(elem.context, {
                 url: 'upload.php' // Dummy, we don't actually want to send the files anywhere
                ,accept: function() { return false } // Stop the Dropzone.js from uploading files
                ,createImageThumbnails: false // We don't want to do this
                ,clickable: true
                ,uploadMultiple: false
            });
            // For some reason, the click events does not bubble up to the dropzone div element, so we bind it ourself
            var el = $(elem.context);
            el.find('b').click(function() { el.click() });
            /* Handle file uploads */
            dropzone.on('addedfile', function(file) {
                // Stop the Dropzone.js from doing anything else
                this.removeFile(file);
                // Now we can save the file to localStorage ourself
                var name = config.localStorageTags.inputFile+$scope.input.tag;
                lsFile.write(name, file, false, function(err) {
                    // Let the inputs-class know we have got the file
                    if (err) $scope.$parent.setInput($scope.input.tag, undefined);
                    else inputs.setValue($scope.input.tag, file.name);
                    $scope.$emit('inputsChanged');
                });
            });
        }
    }}])
})();
