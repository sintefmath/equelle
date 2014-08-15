(function(){
    angular.module('equelleKitchenSink', [ 'ngRoute',
                                           'eqksPageStart',
                                           'eqksPageEditor',
                                           'eqksPageInputs',
                                           'eqksPageRun'
    ])
    /* Multiple views routing */
    .config(['$routeProvider', function($routeProvider) {
        $routeProvider
            .when('/', {
                 templateUrl: 'page-start.html'
                ,controller: 'startController'
            })
            .when('/editor/', {
                 templateUrl: 'page-editor.html'
                ,controller: 'editorController'
            })
            .when('/inputs/', {
                 templateUrl: 'page-inputs.html'
                ,controller: 'inputsController'
            })
            .when('/run/', {
                 templateUrl: 'page-run.html'
                ,controller: 'runController'
            })
    }])
})();
