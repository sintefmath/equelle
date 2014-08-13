(function(){
    angular.module('eqksUi')
    /* Action-button with spinner and progress */
    .directive('button', function() { return {
        restrict: 'E',
        scope: {
            property: '=bindTo',
            expand: '@expand',
            onclick: '&onClick'
        },
        transclude: true,
        template: '<span ng-transclude></span>',
        link: function($scope, elem, attrs, controller) {
            // Make it a nice-bootstrap button
            elem.addClass('btn').addClass('btn-'+$scope.property.class);

            // Create the Ladda-button
            elem.addClass('ladda-button').attr('data-style','expand-'+$scope.expand);
            var l = Ladda.create(elem[0]);

            // Bind the click event
            elem.click(function(event) {
                $scope.$apply(function() {
                    $scope.onclick({ event: event, e: event });
                });
            });

            // Watch for changes in working and progress
            var updateButton = function() {
                if ($scope.property.working) {
                    l.start();
                    if ($scope.property.showProgress) {
                        l.setProgress($scope.property.progress/100);
                    }
                } else {
                    l.stop();
                    // Ladda removes disabled once it stops, so we should reset it
                    if ($scope.property.disabled) {
                        elem.attr('disabled',true);
                    }
                }
            };
            $scope.$watch('property.working', updateButton);
            $scope.$watch('property.progress', updateButton);
            $scope.$watch('property.showProgress', updateButton);

            // Watch for disabled
            $scope.$watch('property.disabled', function() {
                if ($scope.property.disabled) {
                    elem.attr('disabled',true);
                } else if (!$scope.property.working) {
                    // Don't remove disabled if ladda is doing its thing
                    elem.removeAttr('disabled');
                }
            });

            // Watch for class changes
            $scope.$watch('property.class', function(newval, oldval) {
                elem.removeClass('btn-'+oldval).addClass('btn-'+newval);
            });
        }
    }})
})();
