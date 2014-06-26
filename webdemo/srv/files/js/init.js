jQuery(document).ready(function($) {
    /* Input identifier */
    var identifyInputs = function(source) {
        // Inputs on form
        //var input = { name: 'var_name', tag: 'input_tag', type: 'scalar/scalarsequence/indicies', domain: 'single/vertices/edges/faces/cells', subset: 'na/interior/boundary/all/var:var_name' };
        var inputs = [];
        var funcregx = /^\s*([a-z]\w*)\s*=\s*Input([^\(\s]*)\(\s*"(\w*)"(,\s*(.*))?\)/gm;
        var match;
        while (match = funcregx.exec(source)) {
            funcregx.lastIndex = match.index+match[0].length;
            var input = {
                name: match[1],
                tag: match[3]
            };
            switch (match[2]) {
                case "ScalarWithDefault":
                input.type = 'scalar';
                input.domain = 'single';
                break;
                case "CollectionOfScalar":
                input.type = 'scalar';
                break;
                case "DomainSubsetOf":
                input.type = 'indices';
                break;
                case "SequenceOfScalar":
                input.type = 'scalarsequence';
                input.domain = 'single';
                break;
                default:
                inputs = {err: 'Cannot determine type of input'};
            }
            if (input.type && !input.domain) switch (match[5]) {
                case "InteriorCells()":
                input.domain = 'cells';
                input.subset = 'interior';
                break;
                case "BoundaryCells()":
                input.domain = 'cells';
                input.subset = 'boundary';
                break;
                case "AllCells()":
                input.domain = 'cells';
                input.subset = 'all';
                break;
                case "InteriorFaces()":
                input.domain = 'faces';
                input.subset = 'interior';
                break;
                case "BoundaryFaces()":
                input.domain = 'faces';
                input.subset = 'boundary';
                break;
                case "AllFaces()":
                input.domain = 'faces';
                input.subset = 'all';
                break;
                case "InteriorEdges()":
                input.domain = 'edges';
                input.subset = 'interior';
                break;
                case "BoundaryEdges()":
                input.domain = 'edges';
                input.subset = 'boundary';
                break;
                case "AllEdges()":
                input.domain = 'edges';
                input.subset = 'all';
                break;
                case "InteriorVertices()":
                input.domain = 'vertices';
                input.subset = 'interior';
                break;
                case "BoundaryVertices()":
                input.domain = 'vertices';
                input.subset = 'boundary';
                break;
                case "AllVertices()":
                input.domain = 'vertices';
                input.subset = 'all';
                break;
                default:
                var found;
                if (found = _.find(inputs, function(input) { return input.name == match[5]; })) {
                    input.domain = found.domain;
                    input.subset = 'var:'+found.name;
                } else inputs = {err: 'Cannot determine domain of input'};
            }
            if (!input.type || !input.domain) break;
            inputs.push(input);
        }
        return inputs;
    };

	/* Initialize Equelle editor */
	var editor = ace.edit('equelle');
	editor.setTheme('ace/theme/chrome');
	editor.getSession().setMode('ace/mode/javascript');
	editor.setValue('function hello(world) {\n  var equelle = "is great!";\n}');

	/* C++ viewer */
	var ceditor = ace.edit('cpp');
	ceditor.setTheme('ace/theme/chrome');
	ceditor.setReadOnly(true);
	ceditor.getSession().setMode('ace/mode/c_cpp');

	/* Hook up send-to-compiler link */
    var errorMsg = $('div#compiler-error-msg');
    var errorMsgText = errorMsg.children('pre:first');
    var requiredInputs = $('div#required-inputs');
	$('a#send-to-compiler').click(function(e) {
		e.preventDefault();
        var source = editor.getValue();
		$.post('/compiler/', source, function(ret) {
            if (ret.err) {
                ceditor.setValue('');
                errorMsgText.text(ret.err);
                errorMsg.removeClass('hidden');
                requiredInputs.addClass('hidden');
                requiredInputs.html('');
            } else {
                errorMsg.addClass('hidden');
                errorMsgText.text('');
                ceditor.setValue(ret.out);
                ceditor.gotoLine(0,0);

                /* Find the inputs that we need to run this program */
                var inputs = identifyInputs(source);
                if (inputs.length) {
                    requiredInputs.html('This simulator requires the following input files:');
                    var list = $('<ul></ul>').appendTo(requiredInputs);
                    _.each(inputs, function(input) {
                        list.append('<li>'+input.tag+'</li>');
                    });
                } else {
                    requiredInputs.html('This simulator requires no inputs');
                }
                requiredInputs.removeClass('hidden');
            }
		},'json');
	});

    /* Hook up file selector handler */
    $('form#required-input-files input#file-selector').change(function() {
        var files = this.files;
        var reader = new FileReader();
        var readFile = function(i) {
            if (i < files.length) {
                reader.onloadend = function() {
                    var name = files[i].name;
                    name = name.substring(0, name.lastIndexOf('.'));
                    localStorage.setItem('required-file:'+name, reader.result);
                    readFile(++i);
                }
                reader.readAsText(files[i]);
            }
        }
        readFile(0);
    }).click(function() { this.value = null; });
});
