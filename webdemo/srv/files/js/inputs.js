(function(){
    /* This module provides the routines for setting input variables and uploading files */
    angular.module('equelleKitchenSinkInputs', ['equelleKitchenSinkHelpers', 'equelleKitchenSinkEquelleCompile', 'equelleKitchenSinkExecutableCompile'])
    /* The source-scanning routine */
    .directive('eqksInputs', ['$timeout', 'localStorageFile', 'equelleCompiler', 'executableCompiler', function($timeout, localStorageFile, equelleCompiler, executableCompiler) { return {
         restrict: 'A'
        ,controller: function($scope) {
            /* Check the Equelle source has been compiled correctly */
            if (!equelleCompiler.hasCompiled) {
                this.error = 'No Equelle code has been compiled, please <a href="#/editor/">go back</a> and provide code.';
            } else try {
                equelleCompiler.parseInputs();
            } catch (err) { this.error = err }
        }
        ,link: function(scope, elem, attrs, controller) {
            if (controller.error) controller.showError(controller.error);
            /* Tie together upload button with file-input */
            controller.fileUploadButton.click(controller.clickFileInput);
            /* Start compiling the executable file if the result of the C++ source code is not the same as the one we have */
            if (executableCompiler.hasCompiled()) {
                scope.navigation.enableNext();
            } else {
                /* Wait a little, just in case the user jumps back */
                var execCompileWait = $timeout(function() {
                    var compiler = new executableCompiler.Compiler();
                    /* Compilation started */
                    compiler.on('started', function() {
                        scope.navigation.laddaNext.start();
                    });
                    /* Show compilation progress */
                    compiler.on('progress', function(prog) {
                        /* Receives 0-100, Ladda button expects 0-1 */
                        scope.navigation.laddaNext.setProgress(prog/100);
                    });
                    /* Compilation failed from an internal error */
                    compiler.on('failed', function(err) {
                        console.log('Internal executable compilation error:');
                        console.log(err);
                        // TODO: Log these errors somewhere?
                        scope.navigation.laddaNext.setProgress(0);
                        scope.navigation.laddaNext.stop();
                        scope.navigation.disableNext();
                    });
                    /* Compilation was successful */
                    compiler.on('completed', function() {
                        scope.navigation.laddaNext.setProgress(0);
                        scope.navigation.laddaNext.stop();
                        scope.navigation.enableNext();
                    });
                    compiler.compile();
                }, 4000);
                // TODO: Abort compilation if user navigates away from page?
                elem.on('$destroy', function() {
                    $timeout.cancel(execCompileWait);
                });
            }
        }
 
    }}])
    /* List of scalar inputs */
    .directive('eqksInputsScalars', ['equelleCompiler', function(equelleCompiler) { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            var floatregx = /^\d*(?:\.\d*)?$/;
            /* Find single scalar values in simulator */
            var singles = _.filter(equelleCompiler.getInputs(), function(input) { return (input.type=='scalar' && input.domain=='single')});
            var usedTags = [];
            if (singles.length) {
                var el = $(elem.context).append('<h4>Optional scalar values</h4>');
                var list = $('<form class="form-horizontal"></form>').appendTo(el);
                _.each(singles, function(input) {
                    /* Create input element with annotations */
                    var group = $('<div class="form-group has-success"></div>').appendTo(list);
                    group.append('<label for="eqks-inputs-'+input.tag+'" class="control-label col-sm-1">'+input.tag+'</label>');
                    var ingp = $('<div class="col-sm-2 input-group"></div>').appendTo(group);
                    var inel = $('<input type="text" class="form-control" id="'+input.tag+'" placeholder="Default">').appendTo(ingp);
                    ingp.append('<span class="input-group-addon">'+input.default+'</span>');
                    /* Check if default value is set */
                    var localName = 'eqksInputsSingle-'+input.tag;
                    usedTags.push(localName);
                    if (localStorage.hasOwnProperty(localName)) inel.val(localStorage.getItem(localName));
                    /* Bind to input element changes */
                    inel.on('blur keyup', function() {
                        if (floatregx.test(inel.val())) {
                            group.removeClass('has-error').addClass('has-success');
                            /* Save variable for sending to executing server later */
                            if (this.value) localStorage.setItem(localName, parseFloat(this.value));
                            else localStorage.removeItem(localName);
                        } else {
                            group.removeClass('has-success').addClass('has-error');
                            localStorage.removeItem(localName);
                        }
                    });
                });
            }
            /* Cleanup of old scalar values that are not used in this simulator */
            _.each(_.keys(localStorage),function(key) {
                if (_.str.startsWith(key,'eqksInputsSingle-') && !_.contains(usedTags, key)) {
                    localStorage.removeItem(key);
                }
            });
        }
    }}])
    /* List of input files */
    .directive('eqksInputsFiles', ['localStorageFile', 'equelleCompiler', function(localStorageFile, equelleCompiler) { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            /* Find file input values used in simulator */
            var usedTags = [];
            var listItems = {};
            var files = _.reject(equelleCompiler.getInputs(), function(input) { return (input.type=='scalar' && input.domain=='single')});
            if (files.length) {
                /* Create list element with annotations */
                // TODO: A nicer list
                var el = $(elem.context).append('<h4>Required input files</h4>');
                var list = $('<ul class="list-group col-sm-4"></ul>').appendTo(el);
                _.each(files, function(input) {
                    var item = $('<li class="list-group-item list-group-item-danger">'+input.tag+'</li>').appendTo(list);
                    /* Check if file is uploaded*/
                    var localName = 'eqksInputsFile-'+input.tag;
                    usedTags.push(localName);
                    listItems[localName] = item;
                    if (localStorage.hasOwnProperty(localName)) item.removeClass('list-group-item-danger').addClass('list-group-item-success');
                });
                /* Create file-input for reading files */
                var input = $('<input type="file" multiple>').appendTo($('<form class="hide"></form>').appendTo(el));
                input.change(function() {
                    for (var i = 0; i < this.files.length; i++) {
                        var name = this.files[i].name;
                        name = 'eqksInputsFile-'+name.substring(0, name.lastIndexOf('.'));
                        if (_.contains(usedTags, name)) {
                            /* If this file is required, save it to the localStorage */
                            localStorageFile.remove(name);
                            localStorageFile.write(name, this.files[i], function(err) {
                                // TODO: Nicer way of showing the error
                                if (err != null) {
                                    alert('Error reading file "'+this.files[i].name+'"');
                                    listItems[name].removeClass('list-group-item-success').addClass('list-group-item-danger');
                                } else {
                                    listItems[name].removeClass('list-group-item-danger').addClass('list-group-item-success');
                                    // TODO: Some sort of notification when file is re-uploaded
                                }
                           });
                        }
                    }
                });
                /* Bind to controller button */
                controller.clickFileInput = function() { input[0].value = null; input.click() };
            }
            /* Cleanup of old files that are not used in this simulator */
            _.each(localStorageFile.getFileList(),function(key) {
                if (_.str.startsWith(key,'eqksInputsFile-') && !_.contains(usedTags, key)) {
                    localStorageFile.remove(key);
                }
            });
        }
    }}])
    /* File upload button in nav-bar */
    .directive('eqksInputsFileUploadButton', function() { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            /* Bind button to scope to communicate with file-input in form */
            controller.fileUploadButton = $(elem.context);
        }
    }})
    /* Error window */
    .directive('eqksInputsError', function() { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            var el = $(elem.context);
            // Bind window to inputs-controller
            controller.showError = function(text) { el.html(text).removeClass('hide') };
            controller.hideError = function() { el.addClass('hide') };
            // Initially hide window
            el.addClass('hide');
        }
    }})
})();
