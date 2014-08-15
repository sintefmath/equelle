(function(){
    angular.module('eqksUi')
    /* A simple box for showing error messages */
    .directive('eqksUiErrorBox', [function() { return {
        restrict: 'E',
        scope: {
            property: '=bindTo'
        },
        template: '<div class="alert alert-danger" ng-if="property.show" role="alert" ng-bind-html="property.text"></div>',
    }}])
})();
