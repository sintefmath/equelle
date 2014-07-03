(function(){
    angular.module('equelleKitchenSinkEquelleInputs', ['equelleKitchenSinkConfiguration', 'equelleKitchenSinkHelpers'])
    /* Functions that handle the inputs for the Equelle simulator */
    .factory('equelleInputs', ['eqksConfig','localStorageFile', function(eqksConfig,localStorageFile) { 
        /* Read and write inputs list from localStorage */
        var saveInputs = function(inputs) { localStorage.setItem(eqksConfig.localStorageTags.inputs, JSON.stringify(inputs)) };
        var allInputs = function() { if (localStorage.getItem(eqksConfig.localStorageTags.inputs)) return JSON.parse(localStorage.getItem(eqksConfig.localStorageTags.inputs)); else return []; };

        /* Read and write Grid settings */
        var saveGrid = function(grid) {
            var copy = $.extend(true, {}, grid);
            /* Make sure datatypes are correct */
            copy.dimensions = parseInt(copy.dimensions);
            copy.size = _.map(copy.size, function(i) { return parseInt(i) });
            copy.cellSize = _.map(copy.cellSize, function(f) { return parseFloat(f) });
            /* Save new values */
            localStorage.setItem(eqksConfig.localStorageTags.grid, JSON.stringify(copy))
        };
        var readGrid = function() { 
            if (localStorage.getItem(eqksConfig.localStorageTags.grid)) return JSON.parse(localStorage.getItem(eqksConfig.localStorageTags.grid));
            else { localStorage.setItem(eqksConfig.localStorageTags.grid, JSON.stringify(eqksConfig.grid.defaults)); return eqksConfig.grid.defaults }
        };

        /* Pre-declare rest of functions (because they use eachother */
        var parseInputsFromSource, isSingleScalar, setInputValue, getInputValue, inputHasValue;

        /* The function that parses the Equelle code and saves required (and optional) inputs to the localStorage */
        parseInputsFromSource = function() {
            /* Scan the source for input values */
            var inputs = [], match;
            //var funcregx = /^\s*([a-z]\w*)\s*=\s*Input([^\(\s]*)\(\s*"(\w*)"(,\s*(.*))?\)/gm;
            // TODO: Document this!
            var funcregx = /^\s*([a-z]\w*)\s*(?:\:[^\=\n]*\s*)?=\s*Input([^\(\s]*)\(\s*"(\w*)"(?:,\s*(.*))?\)/gm;
            if (localStorage.getItem(eqksConfig.localStorageTags.equelleSource)) while (match = funcregx.exec(localStorage.getItem(eqksConfig.localStorageTags.equelleSource))) {
                funcregx.lastIndex = match.index+match[0].length;
                /* Parse matched Regular Expression */
                var input = {
                    name: match[1],
                    tag: match[3]
                };
                /* Determine type of input */
                switch (match[2]) {
                    case 'ScalarWithDefault': input.type = 'scalar'; input.domain = 'single'; input.default = match[4]; break;
                    case 'CollectionOfScalar': input.type = 'scalar'; break;
                    case 'DomainSubsetOf': input.type = 'indices'; break;
                    case 'SequenceOfScalar': input.type = 'scalarsequence'; input.domain = 'single'; break;
                    default: throw 'Cannot determine type of input with tag: "'+input.tag+'"';
                }
                /* Determine domain of input */
                if (!input.domain) switch (match[4]) {
                    case 'InteriorCells()': input.domain = 'cells'; input.subset = 'interior'; break;
                    case 'BoundaryCells()': input.domain = 'cells'; input.subset = 'boundary'; break;
                    case 'AllCells()': input.domain = 'cells'; input.subset = 'all'; break;
                    case 'InteriorFaces()': input.domain = 'faces'; input.subset = 'interior'; break;
                    case 'BoundaryFaces()': input.domain = 'faces'; input.subset = 'boundary'; break;
                    case 'AllFaces()': input.domain = 'faces'; input.subset = 'all'; break;
                    case 'InteriorEdges()': input.domain = 'edges'; input.subset = 'interior'; break;
                    case 'BoundaryEdges()': input.domain = 'edges'; input.subset = 'boundary'; break;
                    case 'AllEdges()': input.domain = 'edges'; input.subset = 'all'; break;
                    case 'InteriorVertices()': input.domain = 'vertices'; input.subset = 'interior'; break;
                    case 'BoundaryVertices()': input.domain = 'vertices'; input.subset = 'boundary'; break;
                    case 'AllVertices()': input.domain = 'vertices'; input.subset = 'all'; break;
                    default:
                    var found;
                    if (found = _.find(inputs, function(input) { return input.name == match[4]; })) {
                        input.domain = found.domain;
                        input.subset = 'var:'+found.name;
                    } else throw 'Cannot determine domain of input with tag: "'+input.tag+'"';
                }
                inputs.push(input);
            }
            /* Merge our newly parsed inputs with old ones if we have some values set */
            var merged = _.map(inputs, function(input) {
                var oldVal = getInputValue(input.tag);
                if (!!oldVal) input.value = oldVal;
                return input;
            });
            saveInputs(merged);
        };

        /* Functions for determining the type of inputs */
        isSingleScalar = function(input) { return (input.type == 'scalar' && input.domain == 'single') };
        isFile = function(input) { return !isSingleScalar(input); };

        /* Set and retreive values for inputs */
        setInputValue = function(tag, val) {
            var inputs = allInputs();
            if (_.find(inputs, function(input) { return input.tag == tag })) {
                var newInputs = _.map(inputs, function(input) { if (input.tag == tag) { input.value = val }; return input });
                saveInputs(inputs);
            } else throw 'The given input tag was not found in the list of inputs';
        };
        getInputValue = function(tag) {
            var inputs = allInputs();
            var input = _.find(inputs, function(input) { return input.tag == tag });
            if (input && input.value != undefined) {
                if (isFile(input)) {
                    /* If it is a file, check that we have also stored the file for this input */
                    var name = eqksConfig.localStorageTags.inputFile+tag;
                    if (localStorageFile.hasFile(name)) return input.value;
                    else {
                        /* Do some cleanup */
                        setInpuValue(tag, undefined);
                        return undefined;
                    }

                } else return input.value;
            } else return undefined;
        };
        inputHasValue = function(tag) { return !!getInputValue(tag) };

        return {
             parse: parseInputsFromSource
            ,get: allInputs
            ,getSingleScalars: function() { return _.filter(allInputs(), isSingleScalar) }
            ,getFiles: function() { return _.filter(allInputs(), isFile) }
            ,getValue: getInputValue
            ,hasValue: inputHasValue
            ,setValue: setInputValue
            ,getGrid: readGrid
            ,setGrid: saveGrid
    }}])
})();
