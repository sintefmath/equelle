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
            console.log(paramArr);
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
            this.buildParams();
            this.makeFileList();
            /* Make configuration for this simulation */
            var config = {};
            config.signature = localStorage.getItem(eqksConfig.localStorageTags.executableSignature);
            config.name = eqksConfig.executableName;
            config.paramFileName = eqksConfig.paramFileName;
            config.files = _.map(this._fileList, function(file) { return { name: file.name, compressed: file.compressed } });
            /* Send to server and go! */
            console.log(this._paramsBlob);
            console.log(this._requiredInputFiles);
            console.log(this._fileList);
            console.log(config);
        };

        /* Expose classes to outside */
        return {
            Executer: Executer
    }}])
})()
