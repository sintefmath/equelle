(function(){
    angular.module('eqksUi')
    /* Slider component */
    .directive('eqksUiSlider', [function() { return {
        restrict: 'E',
        scope: {
            property: '=bindTo'
        },
        template: '<input type="text">',
        link: function(scope, elem, attrs) {
            // Check that we have necessary values in property
            scope.property.min = scope.property.min || 0;
            scope.property.max = scope.property.max || 0;
            scope.property.value = scope.property.value || 0;

            var tooltip = 'show';
            if (attrs['showTooltip']) switch (attrs['showTooltip']) {
                case 'false':
                    tooltip = 'hide';
                    break;
                case 'always':
                    tooltip = 'always';
                    break;
            }

            // Initialize the slider
            elem.slider({ 
                min: scope.property.min, 
                max: scope.property.max, 
                value: scope.property.value,
                tooltip: tooltip
            });

            // Bind to changes of the property
            scope.$watch('property.min', function() {
                elem.slider('setAttribute','min', scope.property.min).slider('refresh');
            });
            scope.$watch('property.max', function() {
                elem.slider('setAttribute','max', scope.property.max).slider('refresh');
            });
            scope.$watch('property.value', function() {
                elem.slider('setAttribute','value', scope.property.value).slider('refresh');
            });

            // Bind property.value to changes in the slider
            if (attrs['updateOnSlide'] && attrs['updateOnSlide'] == 'true') {
                elem.on('slide', function(event) {
                    scope.$apply(function() {
                        scope.property.value = event.value;
                    });
                });
            }
            elem.on('slideStop', function(event) {
                scope.$apply(function() {
                    scope.property.value = event.value;
                });
            });

            // For some reason, there is a bug in the tooltip hiding... TODO: Has this been fixed?
            if (tooltip == 'hide') {
                elem.parent('div.slider').find('div#tooltip').css('display','none');
            }
        }
    }}])
})();
