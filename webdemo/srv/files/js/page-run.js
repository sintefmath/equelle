(function(){
    angular.module('eqksPageRun', [ 'eqksUi',
                                    'eqksServer',
                                    'eqksXtk'
    ])
    .controller('runController', ['$scope','eqksExecutableRun','$timeout', function($scope,runner,$timeout) {
        /* Header and navigation */
        $scope.title = 'Run:';
        $scope.navigation = {
             previous: { path: '/inputs/', text: 'Provide inputs' }
        };

        /* Run button */
        $scope.runButton = {
            working: false,
            class: 'primary'
        };
        $scope.onRunClick = function(event) {
            if (!$scope.simulatorState.running) {
                runner.run();
            } else {
                runner.abort();
            }
        };

        /* Bind to simulator state */
        $scope.simulatorState = runner.getState();
        runner.on('stateUpdate', function() {
            $scope.runButton.class = ( $scope.simulatorState.running ? 'danger' : 'primary' );
            $scope.$digest();
        });

        /* Bind to runner events */
        runner.on('started', function() {
            console.log('Simluator run STARTED');
        });
        runner.on('failed', function(error) {
            if (error) {
                // Add errors to normal stderr
                $timeout(function() { 
                    $scope.$apply(function() {
                        $scope.simulatorState.output.stderr += '\n\nSimulator run error: '+error.toString();
                    });
                },0);
            }
        });
        runner.on('completed', function() {
            console.log('Simluator run COMPLETED');
        });

        /* Rendering options */
        $scope.$watch('simulatorState.data.grid', function() {
            var grid = $scope.simulatorState.data.grid;
            if (grid) {
                $scope.sliceX = { min: 0, max: grid.size[0]-1, value: Math.floor(grid.size[0]/2) };
                $scope.sliceY = { min: 0, max: grid.size[1]-1, value: Math.floor(grid.size[1]/2) };
                $scope.sliceZ = { min: 0, max: grid.size[2]-1, value: Math.floor(grid.size[2]/2) };
            } else {
                $scope.sliceX = { min: 0, max: 0, value: 0 };
                $scope.sliceY = { min: 0, max: 0, value: 0 };
                $scope.sliceZ = { min: 0, max: 0, value: 0 };
            }
        }, true);

        $scope.renderingOptions = {
            volumeRendering: false,
            lowerThreshold: { min: 0, max: 255, value: 0 },
            upperThreshold: { min: 0, max: 255, value: 255 },
            globalThreshold: false
        };

        $scope.dataTags = [];
        $scope.selectedTag = undefined;
        $scope.selectedData = {
            data: undefined
        };

        var setVisualizationData = function() {
            if ($scope.selectedTag) {
                var set = $scope.selectedTag.sets[$scope.timestep.value];
                if (set) {
                    $scope.selectedData.data = {
                        numbers: set.data,
                        max: ( $scope.renderingOptions.globalThreshold ? $scope.selectedTag.max : set.max ),
                        min: ( $scope.renderingOptions.globalThreshold ? $scope.selectedTag.min : set.min )
                    }
                    return;
                }
            }
            $scope.selectedData.data = undefined;
        };

        /* Selection of tag to use for visualization */
        $scope.$watch('simulatorState.output.data', function() {
            $scope.dataTags = _.keys($scope.simulatorState.output.data);
        }, true);

        /* Timestep slider */
        $scope.timestep = { min: 0, max: 0, value: 0 };
        $scope.$watch('selectedTag', function() {
            if ($scope.selectedTag) {
                $scope.timestep.max = $scope.selectedTag.sets.length-1;
                $scope.timestep.value = Math.min($scope.timestep.value, $scope.timestep.max);
                setVisualizationData();
            } else {
                $scope.timestep.max = 0;
                $scope.timestep.value = 0;
                $scope.selectedData.data = undefined;
            }
        }, true);
        $scope.$watch('timestep.value', function() {
            setVisualizationData();
        });

        /* Global scaling option */
        $scope.$watch('renderingOptions.globalThreshold', function() {
            setVisualizationData();
        });
    }])
})();
