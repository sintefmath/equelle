(function(){
    angular.module('equelleKitchenSinkExecutableRun', ['equelleKitchenSinkConfiguration', 'equelleKitchenSinkHelpers', 'equelleKitchenSinkInputs'])
    /* Classes that take care of sending all neccessary files back to server, then runs the simulator and emits the output */
    .factory('executableRun', ['eqksConfig','equelleInputs','localStorageFile', function(eqksConfig, equelleInputs, localStorageFile) { 
        /* A class that abstracts away the logic of running the executable on the server, extends Events from Backbone */
        Executer = function() { };
        /* Extend with event emitter */
        _.extend(Executer.prototype, Backbone.Events);
        /* A function that creates the .param file we will send to server */
        Executer.prototype.buildParams = function() {
            var paramArr = [], paramAdd = function(p) { paramArr.push(p+'\n') };
            /* Grid dimension and sizes */
            var grid = equelleInputs.getGrid();
            paramAdd('grid_dim='+grid.dimensions);
            switch (grid.dimensions) {
                case 3:
                paramAdd('nz='+grid.size[2]);
                paramAdd('dz='+grid.cellSize[2]);
                case 2:
                paramAdd('ny='+grid.size[1]);
                paramAdd('dy='+grid.cellSize[1]);
                case 1:
                paramAdd('nx='+grid.size[0]);
                paramAdd('dx='+grid.cellSize[0]);
            }
            /* Input values */
            _.each(equelleInputs.getSingleScalars(), function(input) {
                if (input.value) {
                    paramAdd(input.tag+'='+input.value);
                }
            });
            /* Input files */
            var requiredFiles = this._requiredInputFiles = [];
            _.each(equelleInputs.getFiles(), function(input) {
                if (input.value) {
                    paramAdd(input.tag+'_from_file=true');
                    paramAdd(input.tag+'_filename='+input.tag);
                    requiredFiles.push(input.tag);
                }
            });
            // TODO: Handle output to file
            paramAdd('output_to_file=false');
            /* Save the file blob */
            this._paramsBlob = new Blob(paramArr);
        };
        /* A function that builds the list of files we will send to the server */
        Executer.prototype.makeFileList = function() {
            var list = this._fileList = [];
            /* Add the executable and parameters to the list */
            list.push({ name: eqksConfig.executableName, compressed: true, blob: localStorageFile.read(eqksConfig.localStorageTags.executable) });
            list.push({ name: eqksConfig.paramFileName, compressed: false, blob: this._paramsBlob });
            /* Add the provided input files to the list */
            _.each(this._requiredInputFiles, function(file) {
                list.push({ name: file, compressed: false, blob: localStorageFile.read(eqksConfig.localStorageTags.inputFile+file) });
            });
        };
        /* The function that does the actual running of the simulator */
        Executer.prototype.run = function() {
            self = this;
            var triggerEvent = function() { self.trigger.apply(self, arguments) };
            /* Make configuration for this simulation */
            this.buildParams();
            this.makeFileList();
            var config = {};
            config.signature = localStorage.getItem(eqksConfig.localStorageTags.executableSignature);
            config.name = eqksConfig.executableName;
            config.paramFileName = eqksConfig.paramFileName;
            config.files = _.map(this._fileList, function(file) { return { name: file.name, compressed: file.compressed } });
            /* Indicate work beeing done to user */
            triggerEvent('started');
            /* Open connection to server */
            var sock = new WebSocket('ws://'+eqksConfig.compileHost+'/socket/', 'executable-run');
            /* Handle socket errors */
            sock.onerror = function(err) { triggerEvent('failed', err) };
            /* Message protocol */
            var executer = this;
            sock.onmessage = function(msg) {
                if (msg.data instanceof Blob || msg.data instanceof ArrayBuffer) {
                    // TODO: How do we handle output files?
                } else {
                    try {
                        var data = JSON.parse(msg.data);
                        switch (data.status) {
                            /* Ready for us to send the run-configuration */
                            case 'readyForConfig':
                            sock.send(JSON.stringify({ command: 'config', config: config }));
                            break;
                            /* Ready for us to send the files in the list we sent */
                            case 'readyForFiles':
                            _.each(executer._fileList, function(file) {
                                sock.send(file.blob);
                            });
                            break;
                            /* Ready to run */
                            case 'readyToRun':
                            sock.send(JSON.stringify({ command: 'run' }));
                            break;
                            /* Running */
                            case 'running':
                            if (data.stdout) triggerEvent('stdout', data.stdout);
                            if (data.stderr) triggerEvent('stderr', data.stderr);
                            if (data.progress != undefined) triggerEvent('progress', data.progress);
                            break;
                            /* Done */
                            case 'complete':
                            triggerEvent('complete');
                            sock.close();
                            break;
                            case 'failed': throw data.err; break;
                            default: throw ('Unrecognized server status: '+data.status);
                        }
                    } catch (e) { triggerEvent('failed', e); errorTriggered = true; sock.close() }
                }
            };
            /* Socket is closed */
            sock.onclose = function() { };
            // TODO: Error handling here?
        };

        /* Expose classes to outside */
        return {
            Executer: Executer
    }}])
})()
