(function(){
    /* eqksIOStore module with helpers for inputs, outputs and the grid*/
    angular.module('eqksIOStore', [
        'eqksConfiguration',
        'eqksFileHelpers'
    ])

    /* Helpers for inputs to the Equelle program */
    .factory('eqksInputs', ['eqksConfig','localStorageFile', function(config,lsFile) { 
        /* Read and write inputs list from localStorage */
        var write = function(inputs) { localStorage.setItem(config.localStorageTags.inputs, JSON.stringify(inputs)) };
        var read = function() { if (localStorage.getItem(config.localStorageTags.inputs)) return JSON.parse(localStorage.getItem(config.localStorageTags.inputs)); else return []; };

        /* Determining type of inputs */
        var isSingleScalar = function(input) { return input.type == 'Scalar' };
        var isFile = function(input) { return !isSingleScalar(input) };

        /* Set/Get values for sinle input-tags */
        var setInputValue = function(tag, value) {
            var inputs = read();
            var found = _.find(inputs, function(input) { return input.tag == tag });
            if (found) {
                found.value = value;
            } else {
                // Add new input even though it is not needed, this can happen if inputs are set before the source is compiled, i.e. when an example is loaded
                // They will be merged with the correct type and default values later
                inputs.push({
                    tag: tag,
                    value: value
                });
            }
            write(inputs);
        };
        var getInputValue = function(tag) {
            var inputs = read();
            var input = _.find(inputs, function(input) { return input.tag == tag });
            if (input && input.value !== undefined) {
                if (isFile(input)) {
                    // If this input needs a file, check that the file is actually stored in localStorage as well
                    if (lsFile.hasFile(config.localStorageTags.inputFile+tag)) return input.value;
                    else {
                        // Did not find this file, clear value
                        setInputValue(tag, undefined);
                        return undefined;
                    }
                } else return input.value;
            } else return undefined;
        };
        var hasInputValue = function(tag) { return getInputValue(tag) !== undefined }

        /* Parse inputs from compiler-output */
        var parse = function(inputs) {
            // Merge inputs with old values if we already have them stored
            inputs = _.map(inputs, function(input) {
                var oldVal = getInputValue(input.tag);
                if (oldVal !== undefined) input.value = oldVal;
                return input;
            });
            // Save new inputs
            write(inputs);
        };

        /* Expose functions to outside */
        return {
            get: read,
            parse: parse,
            getSingleScalars: function() { return _.filter(read(), isSingleScalar) },
            getFiles: function() { return _.filter(read(), isFile) },
            setValue: setInputValue,
            getValue: getInputValue,
            hasValue: hasInputValue
        };
    }])

    /* Helpers for outputs of the Equelle program */
    .factory('eqksOutputs', ['eqksConfig', function(config) { 
    }])

    /* Helpers for the grid of the Equelle program */
    .factory('eqksGrid', ['eqksConfig', function(config) { 
        /* Write Grid to localStorage */
        var write = function(grid) {
            var defaults = config.grid.defaults;

            // Copy to new object so that we don't change the original, and ensure sizes and datatypes
            var copy = {};
            copy.dimensions = parseInt(grid.dimensions || defaults.dimensions);
            copy.size = _.map(defaults.size, function(s,i) { return parseInt(grid.size[i] || s) });
            copy.cellSize = _.map(defaults.cellSize, function(s,i) { return parseFloat(grid.cellSize[i] || s) });
			copy.abs_res_tol = parseDouble(grid.abs_res_tol || defaults.abs_res_tol);

            // Save copied and defaulted values
            localStorage.setItem(config.localStorageTags.grid, JSON.stringify(copy))
            return copy;
        };

        /* Read grid from localStorage */
        var read = function() {
            // Return previously saved grid
            if (localStorage.getItem(config.localStorageTags.grid)) return JSON.parse(localStorage.getItem(config.localStorageTags.grid));
            // ... or save default values, then return
            else { localStorage.setItem(config.localStorageTags.grid, JSON.stringify(config.grid.defaults)); return read() }
        };

        /* Expose functions to outside */
        return {
            get: read,
            set: write
        };
    }])
})();
