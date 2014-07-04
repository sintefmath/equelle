(function(){
    angular.module('equelleKitchenSinkHelpers', [])
    /* Read and write Blobs to localStorage */
    .factory('localStorageFile', function() {
        var write = function(key, blob, doneCB) {
            if (blob instanceof ArrayBuffer) blob = new Blob([blob]);
            var length = blob.size;
            /* If length is odd number of bytes, we need to pad with a byte at the end so that we don't loose the last one */
            if (length%2) blob = new Blob([blob,new ArrayBuffer(1)]);
            /* Read the bytes into a 16-bit encoded string */
            var fr = new FileReader();
            fr.onloadend = function() { 
                localStorage.setItem(key+'-contents', fr.result);
                localStorage.setItem(key+'-length', length);
                doneCB(fr.error);
            };
            fr.readAsText(blob,'utf-16');
        };
        var read = function(key) {
            var str = localStorage.getItem(key+'-contents');
            var length = parseInt(localStorage.getItem(key+'-length'));
            /* Read all the bytes into a buffer */
            var buf = new ArrayBuffer(str.length*2);
            var bufView = new Uint16Array(buf);
            for ( var i = 0; i < str.length; i++) { bufView[i] = str.charCodeAt(i) }
            /* Create a new blob from the read data, skipping possible padding byte at end */
            return new Blob([buf.slice(0,length)]);
        };
        var remove = function(key) {
            localStorage.removeItem(key+'-contents');
            localStorage.removeItem(key+'-length');
        };
        var getFileList = function() {
            var list = [];
            var keys = _.keys(localStorage);
            var contents = _.map(_.filter(keys, function(key) { return _.str.endsWith(key,'-contents') }), function(key) { return key.slice(0,-9) });
            var lengths = _.map(_.filter(keys, function(key) { return _.str.endsWith(key,'-length') }), function(key) { return key.slice(0,-7) });
            return _.intersection(contents,lengths);
        };
        var removeAll = function() {
            _.each(getFileList(), remove);
        };
        var hasFile = function(key) {
            var keys = _.keys(localStorage);
            return (_.contains(keys, key+'-contents') && _.contains(keys, key+'-length'));
        };
        /* TODO: Remove this debug function */
        window.eqksDownloadFile = function(key) {
            if (hasFile(key)) {
                var blob = read(key);
                var fr = new FileReader();
                fr.onloadend = function() {
                    window.open(fr.result,'_blank');
                };
                fr.readAsDataURL(blob);
            }
        };

        /* Expose functions to outside */
        return {
             write: write
            ,read: read
            ,remove: remove
            ,removeAll: removeAll
            ,getFileList: getFileList
            ,hasFile: hasFile
    }})
})();
