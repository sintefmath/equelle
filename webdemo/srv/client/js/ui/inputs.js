(function(){
    angular.module('eqksUi')
    // TODO: Need support for more browsers and OSX here?
    /* Modify inputs to only allow floats into it */
    .directive('eqksInputFloat', function() { return {
         restrict: 'A'
        ,link: function($scope, elem, attrs) {
            /* Find allowed range */
            var max = parseFloat(attrs.max), min = parseFloat(attrs.min);
            if (isNaN(max)) max = Infinity;
            if (isNaN(min)) min = -Infinity;
            /* Handle inputs to the element */
            var prevValue;
            elem.on('keypress', function(event) {
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers in this box */
                else if (!/[\-0-9.]/.test(ch)) event.preventDefault();
                /* Only allow one . */
                else if (ch == '.' && _.str.contains(this.value,'.')) event.preventDefault();
                /* Only allow one - at the very beginning */
                else if (ch == '-' && this.selectionStart != 0) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum and under minimum */
                var f = parseFloat(this.value);
                if (f > max || f < min) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                /* Only allow pasting if it is a float */
                if (isNaN(parseFloat(text))) event.preventDefault();
            });
            /* Call update function when a change is made */
            if (attrs.eqksInputChanged) {
                elem.on('change', function() {
                    var val;
                    // Check if the value is valid
                    if (elem.val().match(/^-?\d*(.\d*)?$/)) {
                        val = parseFloat(elem.val());
                        if (isNaN(val)) val = undefined;
                        else if (val > max || val < min) val = undefined;
                    }
                    // Clear text field is something invalid is there
                    if (val === undefined) elem.val('');

                    $scope.$eval(attrs.eqksInputChanged, { v: val, val: val, value: val });
                });
            }
            /* Check that initial value is acceptable */
            if (attrs.eqksInputValue) {
                var val = $scope.$eval(attrs.eqksInputValue);
                if (val === undefined) {
                    elem.val('');
                } else {
                    val = parseFloat(val);
                    if (isNaN(val) || val > max || val < min) {
                        elem.val('');
                        elem.trigger('change');
                    } else {
                        elem.val(val);
                    }
                }
            }
        }
    }})

    /* Modify inputs to only allow positive floats into it */
    .directive('eqksInputPositiveFloat', function() { return {
         restrict: 'A'
        ,require: '^eqksInputs'
        ,link: function(scope, elem, attrs, controller) {
            /* Find allowed range */
            var max = parseFloat(attrs.max);
            if (isNaN(max)) max = Infinity;
            /* Handle inputs to the element */
            var prevValue;
            elem.on('keypress', function(event) {
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers and . */
                else if (!/[0-9.]/.test(ch)) event.preventDefault();
                /* Only allow one . */
                else if (ch == '.' && _.str.contains(this.value,'.')) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum */
                if (parseFloat(this.value) > max) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                var f = parseFloat(text);
                /* Only allow pasting if it is a positive float */
                if (isNaN(f) || f <= 0) event.preventDefault();
            });
            /* Call update function when a change is made */
            if (attrs.eqksInputChanged) {
                elem.on('change', function() {
                    var val;
                    // Check if the value is valid
                    if (elem.val().match(/^\d*(.\d*)?$/)) {
                        val = parseFloat(elem.val());
                        if (isNaN(val)) val = undefined;
                        if (val <= 0 || val > max) val = undefined;
                    }
                    // Clear text field is something invalid is there
                    if (val === undefined) elem.val('');

                    $scope.$eval(attrs.eqksInputChanged, { v: val, val: val, value: val });
                });
            }
            /* Check that initial value is acceptable */
            if (attrs.eqksInputValue) {
                var val = $scope.$eval(attrs.eqksInputValue);
                if (val === undefined) {
                    elem.val('');
                } else {
                    val = parseFloat(val);
                    if (isNaN(val) || val > max || val <= 0) {
                        elem.val('');
                        elem.trigger('change');
                    } else {
                        elem.val(val);
                    }
                }
            }
        }
    }})

    /* Modify inputs to only allow positive integers into it */
    .directive('eqksInputsPositiveInteger', function() { return {
         restrict: 'A'
        ,link: function(scope, elem, attrs) {
            /* Find allowed range */
            var max = parseInt(attrs.max);
            if (isNaN(max)) max = Infinity;
            /* Handle inputs to the element */
            var prevValue;
            elem.on('keypress', function(event) {
                prevValue = this.value;
                var ch = String.fromCharCode(event.which);
                /* Allow copy/cut/paste */
                if (event.ctrlKey) return;
                /* Only allow numbers in this box */
                else if (!/[0-9]/.test(ch)) event.preventDefault();
            }).on('input', function() {
                /* Stop value over maximum */
                if (parseInt(this.value) > max) {
                    this.value = prevValue;
                    elem.trigger('change');
                }
            }).on('paste', function(event) {
                var text = (event.originalEvent || event).clipboardData.getData('text/plain');
                var i = parseInt(text);
                /* Only allow pasting if it is a positive integer */
                if (isNaN(i) || i < 1 || i > max) event.preventDefault();
            });
            /* Call update function when a change is made */
            if (attrs.eqksInputChanged) {
                elem.on('change', function() {
                    var val;
                    // Check if the value is valid
                    if (elem.val().match(/^\d+$/)) {
                        val = parseInt(elem.val());
                        if (isNaN(val)) val = undefined;
                        if (val < 1 || val > max) val = undefined;
                    }
                    // Clear text field is something invalid is there
                    if (val === undefined) elem.val('');

                    $scope.$eval(attrs.eqksInputChanged, { v: val, val: val, value: val });
                });
            }
            /* Check that initial value is acceptable */
            if (attrs.eqksInputValue) {
                var val = $scope.$eval(attrs.eqksInputValue);
                if (val === undefined) {
                    elem.val('');
                } else {
                    val = parseInt(val);
                    if (isNaN(val) || val > max || val < 1) {
                        elem.val('');
                        elem.trigger('change');
                    } else {
                        elem.val(val);
                    }
                }
            }
        }
    }})
})();
