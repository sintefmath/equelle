(function(){
    angular.module('eqksUi')
    /* Simple wrapper for bootstrap-tabbing */
    .directive('eqksTabbing', function() { return {
        restrict: 'A',
        link: function($scope, elem, attrs, controller) {
            // Activate tabbing on links
            if (elem.is('ul.nav-tabs')) {
                elem.find('li a').click(function(event) {
                    event.preventDefault();
                    var a = $(this);
                    if (!a.parent('li').hasClass('disabled')) {
                        a.tab('show');
                    }
                });
            }
        }
    }})
})();
