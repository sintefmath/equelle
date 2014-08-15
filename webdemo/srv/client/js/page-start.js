(function(){
    angular.module('eqksPageStart', [ 'eqksUi',
                                      'eqksConfiguration',
                                      'eqksIOStore',
                                      'eqksFileHelpers'
    ])
    /* The Start-page controller */
    .controller('startController', ['$scope','$http','localStorageFile','eqksConfig','eqksInputs','eqksGrid', function($scope, $http, lsFile, config, inputs, grid) {
        /* Header and navigation */
        $scope.title = 'Select a project:';
        $scope.navigation = {};

        /* Examples */
        $scope.simulators = [
            { id: ':empty', name: 'Empty', text: 'Start with no code or data' }
        ];
        // Append loaded examples
        if (equelleKitchenSinkExamples && equelleKitchenSinkExamples.examples) _.each(equelleKitchenSinkExamples.examples, function(example) {
                var ex = { id: example.id, name: example.name };
                if (example.description) ex.text = example.description;
                if (example.image) ex.imgSrc = example.image;
                $scope.simulators.push(ex);
        });
        // Check if we can resume previous work
        if (localStorage.getItem(config.localStorageTags.equelleSource)) {
            var prev = { id: ':resume', name: 'Resume previous', text: 'Resume with the previous simulator you were working on' };
            var img = localStorage.getItem(config.localStorageTags.previousThumbnail);
            if (img) prev.imgSrc = img;
            $scope.simulators.push(prev);
        }

        /* Error messages */
        $scope.error = {
            show: false,
            text: ''
        };

        /* Set localStorage data according to selected project */
        // Function for clearing all localStorage data set by this app
        var clearLocalStorage = function() {
            _.each(_.keys(localStorage), function(key) {
                if (_.str.startsWith(key, config.localStorageTags.prefix)) localStorage.removeItem(key);
            });
        };
        // The setSimulator function
        $scope.setSimulator = function(e,id) {
            // Clear old errors
            $scope.error.show = false;
            $scope.error.text = '';

            if (id== ':resume') {} // Don't do anything, all previous data should be in localStorage
            else if (id == ':empty') {
                clearLocalStorage();
                localStorage.setItem(config.localStorageTags.equelleSource,'');
            } else {
                clearLocalStorage();
                e.preventDefault();
                // Load project data from example files
                $http.get('/examples/'+id)
                .success(function(data) {
                    // Set the source code
                    localStorage.setItem(config.localStorageTags.equelleSource,data.source);
                    // Set grid options
                    grid.set(data.grid);
                    // Load input-files
                    var done = _.after(data.inputfiles.length, function() {
                        /* Finally, move to the editor page */
                        var link = e.currentTarget.href;
                        window.location.href = link;
                    });
                    _.each(data.inputfiles, function(file) {
                        $http.get('/examples/'+id+'/'+file.name, { responseType: 'blob' })
                        .success(function(data) {
                            // Save file
                            lsFile.writeRaw(config.localStorageTags.inputFile+file.tag, data, file.compressed, done)
                            inputs.setValue(file.tag, file.name);
                        })
                        .error(done);
                    });
                })
                .error(function(data, status) {
                    // If an error occured during loading, show it
                    $scope.error.text = '<strong>Error while loading example:</strong> '+status;
                    $scope.error.show = true;
                });
            }
        };
    }])
})();
