(function(){
    /* This module provides the title, navigation and actions bar on top */
    angular.module('equelleKitchenSinkNavigation', [])
    /* Title and navigation bar */
    .directive('eqksHeaderNavigation', function() { return {
         restrict: 'E'
        ,template: '<h2 class="clearfix">{{title}}<div class="btn-toolbar pull-right" role="toolbar"><div class="btn-group" ng-transclude></div><div class="btn-group eqks-navigation-buttons"></div></div></h2>'
        ,transclude: true
        ,link: function(scope, elem, attrs) {
            var group = $(elem.context).find('div.eqks-navigation-buttons');
            var navFuncs = {};
            /* Create link-buttons and bind enable/disable functions */
            if (scope.navigation.previous) {
                var nav = scope.navigation.previous;
                var link = $('<a href="#'+nav.path+'" class="btn btn-default" '+(nav.disabled?'disabled':'')+'><span class="glyphicon glyphicon-arrow-left"></span> '+nav.text+'</a>').appendTo(group);
                navFuncs.enablePrevious = function() { link.removeAttr('disabled') };
                navFuncs.disablePrevious = function() { link.attr('disabled') };
            } else {
                navFuncs.enablePrevious = function() { };
                navFuncs.disablePrevious = function() { };
            }
            if (scope.navigation.next) {
                var nav = scope.navigation.next;
                var link = $('<a href="#'+nav.path+'" class="btn btn-default" '+(nav.disabled?'disabled':'')+'>'+nav.text+' <span class="glyphicon glyphicon-arrow-right"></span></a>').appendTo(group);
                navFuncs.enableNext = function() { link.removeAttr('disabled') };
                navFuncs.disableNext = function() { link.attr('disabled') };
            } else {
                navFuncs.enableNext = function() { };
                navFuncs.disableNext = function() { };
            }
            /* Bind enable/disable functions to current scope */
            scope.navigation = navFuncs;
        }
    }})
})();
