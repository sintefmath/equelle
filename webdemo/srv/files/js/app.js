(function(){
    angular.module('equelleKitchenSink', ['ngRoute','equelleKitchenSinkNavigation','equelleKitchenSinkEditor','equelleKitchenSinkInputs','equelleKitchenSinkRun','equelleKitchenSinkHelpers'])
    /* Multiple views routing */
    .config(function($routeProvider) {
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
    })
    /* Page controllers */
    .controller('startController', ['$scope', 'localStorageFile', function($scope, localStorageFile) {
        /* Header and navigation */
        $scope.title = 'Select a project:';
        $scope.navigation = {};
        /* Projects */
        $scope.simulators = [
             { path: ':empty', name: 'Empty', text: 'Start with no code or data' }
            ,{ path: '/examples/heateq/', name: '2D Heat equation', text: 'Start with a simple planar heat equation example' }
            ,{ path: ':resume', name: 'Resume previous', text: 'Resume with the previous simulator you were working on' }
        ];
        $scope.setSimulator = function(e,path) {
            if (path == ':resume') {} // Don't do anything, all previous data should be in localStorage
            else if (path == ':empty') {
                // Clear all data from previous project from localStorage
                localStorage.eqksSource = '';
                localStorage.eqksCompiled = '';
                localStorage.eqksCompiledSign = '';
                localStorage.eqksExecutableSign = '';
                localStorage.eqksExecutableCompiledSign = '';
                localStorageFile.removeAll();
            } else {
                // Load project data from example files
                if (path == '/examples/heateq/') {
                    localStorage.eqksSource = window.equellecode;
                    localStorage.eqksCompiled = '';
                    localStorage.eqksCompiledSign = '';
                    localStorage.eqksExecutableSign = '';
                    localStorage.eqksExecutableCompiledSign = '';
                    localStorageFile.removeAll();
                } else {
                    e.preventDefault();
                }
            }
        };
    }])
    .controller('editorController', function($scope) {
        /* Header and navigation */
        $scope.title = 'Edit code:';
        $scope.navigation = {
             previous: { path: '/', text: 'Select project' }
            ,next: { path: '/inputs/', text: 'Provide inputs', disabled:true }
        };
    })
    .controller('inputsController', function($scope) {
        /* Header and navigation */
        $scope.title = 'Provide inputs:';
        $scope.navigation = {
             previous: { path: '/editor/', text: 'Edit code' }
            ,next: { path: '/run/', text: 'Run simulation', disabled:true, ladda:true }
        };
    })
    .controller('runController', function($scope) {
        /* Header and navigation */
        $scope.title = 'Run:';
        $scope.navigation = {
             previous: { path: '/inputs/', text: 'Provide inputs' }
        };
    })
})();













window.equellecode = '# Heat conduction with Diriclet boundary conditions.\r\n# Heat diffusion constant.\r\nk : Scalar = InputScalarWithDefault(\"k\", 0.3)\r\n\r\n# Compute interior transmissibilities.\r\nifaces = InteriorFaces()     # Deduced type:  Collection Of Face (On itself)\r\nfirst = FirstCell(ifaces)    # Deduced type:  Collection Of Cell On ifaces\r\nsecond = SecondCell(ifaces)  # Deduced type:  Same as for \'first\'.\r\nitrans : Collection Of Scalar On ifaces\r\nitrans = k * |ifaces| \/ |Centroid(first) - Centroid(second)| \r\n\r\n# Compute flux for interior faces.\r\ncomputeInteriorFlux : Function(u : Collection Of Scalar On AllCells())\r\n                                -> Collection Of Scalar On InteriorFaces()\r\ncomputeInteriorFlux(u) = {\r\n    -> -itrans * Gradient(u)\r\n}\r\n\r\n# Support for Dirichlet boundaries\r\ndirichlet_boundary : Collection Of Face Subset Of BoundaryFaces()\r\ndirichlet_boundary = InputDomainSubsetOf(\"dirichlet_boundary\", BoundaryFaces())\r\ndirichlet_val : Collection Of Scalar On dirichlet_boundary\r\ndirichlet_val = InputCollectionOfScalar(\"dirichlet_val\", dirichlet_boundary)\r\n\r\n# Compute boundary transmissibilities and orientations.\r\nbf = BoundaryFaces()\r\nbf_cells = IsEmpty(FirstCell(bf)) ? SecondCell(bf) : FirstCell(bf)\r\nbf_sign = IsEmpty(FirstCell(bf)) ? (-1 Extend bf) : (1 Extend bf)\r\nbtrans = k * |bf| \/ |Centroid(bf) - Centroid(bf_cells)|\r\n\r\n# Compute flux for boundary faces.\r\ndir_cells = bf_cells On dirichlet_boundary\r\ndir_sign = bf_sign On dirichlet_boundary\r\ndir_trans = btrans On dirichlet_boundary\r\ncomputeBoundaryFlux : Function(u : Collection Of Scalar On AllCells()) ...\r\n                                -> Collection Of Scalar On BoundaryFaces()\r\ncomputeBoundaryFlux(u) = {\r\n    # Compute flux at Dirichlet boundaries.\r\n    u_dirbdycells = u On dir_cells\r\n    dir_fluxes = dir_trans * dir_sign * (u_dirbdycells - dirichlet_val)\r\n    # Extending with zero away from Dirichlet boundaries,\r\n    # which means assuming no-flow elsewhere.\r\n    -> dir_fluxes Extend BoundaryFaces()\r\n}\r\n\r\n# Compute the residual for the heat equation.\r\nvol = |AllCells()|   # Deduced type:  Collection Of Scalar On AllCells()\r\ncomputeResidual : Function(u  : Collection Of Scalar On AllCells(), ...\r\n                           u0 : Collection Of Scalar On AllCells(),\r\n                           dt : Scalar)\r\n                             -> Collection Of Scalar On AllCells()\r\ncomputeResidual(u, u0, dt) = {\r\n    ifluxes = computeInteriorFlux(u)\r\n    bfluxes = computeBoundaryFlux(u)\r\n    # Extend both ifluxes and bfluxes to AllFaces() and add to get all fluxes.\r\n    fluxes = (ifluxes Extend AllFaces()) + (bfluxes Extend AllFaces())\r\n    # Deduced type of \'residual\': Collection Of Scalar On AllCells()\r\n    residual = u - u0 + (dt \/ vol) * Divergence(fluxes)\r\n    -> residual\r\n}\r\n\r\n# u_initial is user input (u is the unknown, temperature here)\r\nu_initial : Collection Of Scalar On AllCells()\r\nu_initial = InputCollectionOfScalar(\"u_initial\", AllCells())\r\n\r\n# Sequences are ordered, and not associated with the grid\r\n# as collections are.\r\ntimesteps : Sequence Of Scalar\r\ntimesteps = InputSequenceOfScalar(\"timesteps\")\r\n\r\n# u0 must be declared Mutable, because we will change it\r\n# in the For loop further down.\r\nu0 : Mutable Collection Of Scalar On AllCells()\r\nu0 = u_initial\r\n\r\nFor dt In timesteps {\r\n    computeResidualLocal : Function(u : Collection Of Scalar On AllCells())\r\n                                     -> Collection Of Scalar On AllCells()\r\n    computeResidualLocal(u) = {\r\n        -> computeResidual(u, u0, dt)\r\n    }\r\n    u_guess = u0\r\n    u = NewtonSolve(computeResidualLocal, u_guess)\r\n    Output(\"u\", u)\r\n    Output(\"maximum of u\", MaxReduce(u))\r\n    u0 = u\r\n}';
