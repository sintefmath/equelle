(function(){
    /* This module provides the routines for setting input variables and uploading files */
    angular.module('equelleKitchenSinkRun', [])
    /* Connect to the server socket */
    .directive('eqksRun', ['$timeout', 'localStorageFile', function($timeout,localStorageFile) { return {
         restrict: 'A'
        ,controller: function($scope) {
        }
        ,link: function(scope, elem, attrs, controller) {
            /* Connect to compiler server */
            var sock = new WebSocket('ws://localhost:8080/socket/', 'executable-run');
            /* Sending file buffers helper */
            var sendStringBytes = function(str) { // str: Must be a binary file converted to a UTF-16 encoded string
                var buf = new ArrayBuffer(str.length*2);
                var bufView = new Uint16Array(buf);
                for ( var i = 0; i < str.length; i++) { bufView[i] = str.charCodeAt(i) }
                var blob = new Blob([buf]);
                console.log(blob);
                sock.send(blob);
            };
            sock.onerror = function(err) { console.log('Socket error:'); console.log(err); };
            sock.onclose = function(code) { console.log('Socket close:'); console.log(code); };
            /* Protocol */
            var sendFiles = [];
            var params;
            sock.onmessage = function(mess) {
                var data = JSON.parse(mess.data);
                console.log('Received message: ');
                console.log(data);
                switch (data.status) {
                    case 'readyForConfig':
                    /* Send a runtime configuration object */
                    var config = { sign: localStorage.eqksExecutableSign, fileList: [] };
                    /* Send executable file */
                    config.fileList.push({name: 'simulator', compressed: true});
                    /* Send input files */
                    _.each(localStorageFile.getFileList(), function(key) { 
                        if (_.str.startsWith(key,'eqksInputsFile-')) {
                            var name = key.substring(15);
                            sendFiles.push(key);
                            config.fileList.push({name: name, compressed: false});
                        }
                    });
                    /* Send params file */
                    config.fileList.push({name: 'params.param', compressed: false});
                    var paramsArr = [];
                    paramsArr.push('grid_dim=2\n');
                    paramsArr.push('nx=10\n');
                    paramsArr.push('ny=10\n');
                    paramsArr.push('dx=0.1\n');
                    paramsArr.push('dy=0.1\n');
                    paramsArr.push('u_initial_from_file=true\n');
                    paramsArr.push('u_initial_filename=u_initial\n');
                    paramsArr.push('timesteps_from_file=true\n');
                    paramsArr.push('timesteps_filename=timesteps\n');
                    params = new Blob(paramsArr);
                    /* Ready to go */
                    sock.send(JSON.stringify({command: 'config', config: config}));
                    console.log('Sent config');
                    console.log(config);
                    break;
                    case 'readyForFiles':
                    console.log('Sending files to server');
                    /* Send all the files one after another */
                    sock.send(localStorageFile.read('eqksExecutable'));
                    _.each(sendFiles.reverse(), function(fileKey) { sock.send(localStorageFile.read(fileKey)) });
                    sock.send(params);
                    console.log('Sent all files to server');
                    break;
                    case 'readyToRun':
                    console.log('Server is ready to run');
                    sock.send(JSON.stringify({command: 'run'}));
                    break;
                    //sendStringBytes(localStorage.eqksExecutable);
                    //case 'success':
                    //sock.close();
                    //break;
                    //case 'failed':
                    //internalError(data.err);
                    //break;
                    default:
                    internalError('Unrecognized status: '+data.status);
                }
            };
        }
    }}])
    /* Output window */
    .directive('eqksRunOutput', function() { return {
         restrict: 'A'
        ,require: '^eqksRun'
        ,link: function(scope, elem, attrs, controller) {
            /* Link output window to controller so that it can use it */
            controller.outWindow = $(elem.context);
        }
    }})
})();
