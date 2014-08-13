(function(){
    angular.module('eqksUi')
    /* A wrapper for the CodeMirror editor */
    .directive('eqksCodemirror', ['equelleCodemirrorHints', function(equelleHints) { return {
         restrict: 'E'
        ,scope: {
            property: '=editor'
         }
        ,template: '<div class="eqks-codemirror-container"></div><eqks-ui-error-box bind-to="error"></eqks-ui-error-box>'
        ,link: function($scope, elem, attrs, controller) {
            // Error messages
            $scope.error = {
                show: false,
                text: ''
            };

            // Create the CodeMirror editor
            var editor = CodeMirror(elem.children('div')[0], {
                 value: $scope.property.source
                ,mode: 'equelle'
                ,lineNumbers: true
                ,gutters: ['equelle-gutter-error']
                ,matchBrackets: true
                ,autoCloseBrackets: { pairs: '()[]{}""||', explode: '{}' }
                ,tabSize: 4
                ,indentUnit: 4
                ,hintOptions: {
                    hint: equelleHints.hint
                }
                ,extraKeys: { 'Ctrl-Space': 'autocomplete' }
            });

            // Watch for changes in the source from outside
            var triggerChanged = true, triggerWatch = true;
            $scope.$watch('property.source', function() {
                if (triggerWatch) {
                    triggerChanged = false;
                    // Clear old errors
                    editor.clearGutter('equelle-gutter-error');
                    $scope.error.show = false;
                    // Set new source
                    editor.setValue($scope.property.source);
                } else {
                    triggerWatch = true;
                }
            });

            // Watch for changes in the source from editor
            editor.on('change', function() {
                if (triggerChanged) {
                    $scope.$apply(function() {
                        triggerWatch = false;
                        $scope.property.source = editor.getValue();
                    });
                } else {
                    triggerChanged = true;
                }
            });

            // Display errors in editor
            var errorElement = $('<span class="glyphicon glyphicon-remove-sign equelle-gutter-error-sign text-danger"></span>')[0];
            var errorLines = [];
            $scope.$watch('property.errors', function() {
                // Clear old errors
                editor.clearGutter('equelle-gutter-error');
                $scope.error.show = false;
                $scope.error.text = '';
                _.each(errorLines, function(line) {
                    editor.removeLineClass(line, 'background', 'bg-danger');
                });

                // Insert new errors
                _.each($scope.property.errors, function(error) {
                    editor.setGutterMarker(error.line, 'equelle-gutter-error', errorElement);
                    $scope.error.text += error.text+'<br>';
                    // Mark all relevant lines with background
                    var state, line = error.line;
                    do {
                        editor.addLineClass(line, 'background', 'bg-danger');
                        errorLines.push(line);
                        state = editor.getStateAfter(line, true);
                        line--;
                    } while (line > 0 && (state.lineTokens.length == 0 || _.last(state.lineTokens).line > 0));
                });

                // Show if there were any
                if ($scope.property.errors.length > 0) {
                    $scope.error.show = true;
                    editor.scrollIntoView({line: $scope.property.errors[0].line, ch: 0}, 100);
                }
            });
        }
    }}])
})();
