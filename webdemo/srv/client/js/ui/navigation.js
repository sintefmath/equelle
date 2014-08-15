(function(){
    angular.module('eqksUi')
    /* Title, navigation and action buttons bar */
    .directive('eqksHeaderNavigation', function() { return {
         restrict: 'E'
        ,templateUrl: '/html/ui/navigation.html'
        ,transclude: true
    }})
})();
