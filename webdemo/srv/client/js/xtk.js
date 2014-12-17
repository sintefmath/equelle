(function(){
    /* Some helper functions for checking and transforming data */
    // Rescale data to 1-255 range within given min-max range (0 is transparent, so don't use that)
    var rescaleData = function(nums, min, max) {
        var scaled = new ArrayBuffer(nums.length);
        var view = new Uint8ClampedArray(scaled);
        /* Scale all numbers */
        if (max != min) {
            var fact = 254/(max-min);
            for (var i = nums.length; i--; ) {
                // Decrementing loops are for some reason faster in JS
                view[i] = Math.round((nums[i]-min)*fact)+1;
            }
        } else {
            for (var i = nums.length-1; i--; ) {
                view[i] = 1;
            }
        }
        return scaled;
    };

    angular.module('eqksXtk', ['eqksIOStore'])
    /* The rendering canvas */
    .directive('eqksXtk', ['eqksInputs', function(equelleInputs) { return {
         restrict: 'A'
        ,scope: {
            data: '=bindData',
            grid: '=',
            sliceX: '=',
            sliceY: '=',
            sliceZ: '=',
            renderingOptions: '='
         }
        ,link: function($scope, elem, attrs, controller) {
            /* Get the grid we are simulating on */
            var grid = $scope.grid;
            /* Check that we can show something on this grid type (intentional fallthrough) */
            var plotType, dimensions;
            switch (grid.dimensions) {
                case 3:
                if (grid.size[0] > 1 && grid.size[1] > 1 && grid.size[2] > 1) {
                    plotType = 1;
                    dataLength = grid.size[0]*grid.size[1]*grid.size[2];
                    dimensions = _.clone(grid.size);
                    break;
                }
                case 2:
                if (grid.size[0] > 1 && grid.size[1] > 1) {
                    plotType = 0;
                    dataLength = grid.size[0]*grid.size[1];
                    dimensions = [grid.size[0], grid.size[1], 1];
                    break;
                } else if (grid.size[0] > 1 && grid.size[2] > 1) {
                    plotType = 0;
                    dataLength = grid.size[0]*grid.size[2];
                    dimensions = [grid.size[0], grid.size[2], 1];
                    break;
                } else if (grid.size[1] > 1 && grid.size[2] > 1) {
                    plotType = 0;
                    dataLength = grid.size[1]*grid.size[2];
                    dimensions = [grid.size[1], grid.size[2], 1];
                    break;
                }
                default:
                console.log('Cannot visualize this grid structure');
                return;
            }
            var dataLength = _.reduce(dimensions, function(m,d) { return m*d }, 1);


            /* Create the function that initializes a rendering context and loads the supplied data */
            var renderer, volume, cameraView;
            var zoomLevel = 0;
            var displayData = function(data) {
                /* If there already is a context */
                if (renderer) {
                    /* Store the old camera position, so the view looks the same in the new one */
                    cameraView = renderer.camera.view;
                    /* Destroy the old one */
                    renderer.destroy();
                    renderer = undefined;
                }
                /* If empty data, nothing to display */
                if (!data) return;
                /* Create the new context */
                switch (plotType) {
                    case 0:
                    /* 2D Grid of cells */
                    renderer = new X.renderer2D();
                    renderer.container = elem.context;
                    renderer.orientation = 'z';
                    renderer.init();
                    renderer.interactor.onMouseWheel = function(event) {
                        /* Mouse wheel up */
                        if (event.detail < 0) {
                            zoomLevel += 1;
                            renderer.camera.zoomIn(false);
                        }
                        /* Mouse wheel down */
                        else if (event.detail > 0) {
                            zoomLevel -= 1;
                            renderer.camera.zoomOut(false);
                        }
                        /* Don't do anything else */
                        event.preventDefault();
                    };
                    window.renderer = renderer;
                    window.interactor = renderer.interactor;
                    break;
                    case 1:
                    /* 3D Grid of cells */
                    renderer = new X.renderer3D();
                    renderer.container = elem.context;
                    renderer.init();
                    break;
                }
                /* Insert volume with provided data */
                volume = new X.volume();
                volume.dimensions = dimensions;
                volume.file = 'data.raw';
                volume.filedata = data;

                /* Colormap */
                volume.colormap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

                /* Set correct slice */
                volume.indexX = $scope.sliceX.value;
                volume.indexY = $scope.sliceY.value;
                volume.indexZ = $scope.sliceZ.value;
                /* Bind to slider changes */
                $scope.$watch('sliceX.value', function() { volume.indexX = $scope.sliceX.value });
                $scope.$watch('sliceY.value', function() { volume.indexY = $scope.sliceY.value });
                $scope.$watch('sliceZ.value', function() { volume.indexZ = $scope.sliceZ.value });

                /* Volume rendering */
                volume.volumeRendering = $scope.renderingOptions.volumeRendering;
                $scope.$watch('renderingOptions.volumeRendering', function() { volume.volumeRendering = $scope.renderingOptions.volumeRendering });

                /* Thresholding */
                volume.lowerThreshold = $scope.renderingOptions.lowerThreshold.value;
                $scope.$watch('renderingOptions.lowerThreshold.value', function() { volume.lowerThreshold = $scope.renderingOptions.lowerThreshold.value });
                volume.upperThreshold = $scope.renderingOptions.upperThreshold.value;
                $scope.$watch('renderingOptions.upperThreshold.value', function() { volume.upperThreshold = $scope.renderingOptions.upperThreshold.value });

                /* Restore camera position */
                if (cameraView) renderer.camera.view = cameraView;
                /* Render */
                renderer.add(volume);
                renderer.render();
                /* Zoom to the correct level */
                renderer.onRender = _.once(function() {
                    if (zoomLevel > 0) _.times(zoomLevel, function() { renderer.camera.zoomIn(false) });
                    else if (zoomLevel < 0) _.times(-zoomLevel, function() { renderer.camera.zoomOut(false) });
                });
            };

            /* Let the user change data */
            $scope.$watch('data', function() {
                if ($scope.data.data) {
                    var rescaled = rescaleData($scope.data.data.numbers, $scope.data.data.min, $scope.data.data.max);

                    if (!rescaled|| !(rescaled instanceof ArrayBuffer) || rescaled.byteLength != dataLength) {
                        console.log('Wrong data size, not rendering data!');
                        //TODO: Warn user here?
                    } else {
                        displayData(rescaled);
                    }
                } else {
                    displayData(undefined);
                }

            }, true);
        }
    }}])
})()
