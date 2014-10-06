(function(){
    angular.module('eqksFileHelpers', [])
    /* Read and write Blobs to localStorage */
    .factory('localStorageFile', function() {
        /* Compression of byte-array using GZip */
        var compress = function(arrbuf) {
            var inview = new Uint8Array(arrbuf);
            var gzip = new Zlib.Gzip(inview);
            var compressed = gzip.compress();
            var outbuf = new ArrayBuffer(compressed.length);
            var outview = new Uint8Array(outbuf);
            outview.set(compressed);
            return outbuf;
        };

        /* Write file contents to localStorage */
        var writeRaw = function(key, blob, compressed, doneCB) {
            var fr = new FileReader();
            var length = blob.size;
            /* If length is odd number of bytes, we need to pad with a byte at the end so that we don't loose the last one */
            if (length%2) blob = new Blob([blob, new ArrayBuffer(1)]);
            /* Read the bytes into a 16 bit encoded string */
            fr.onloadend = function() { 
                if (!fr.error) {
                    // Convert the bytes to a string
                    var view = new Uint16Array(fr.result);
                    var str = '';
                    for (var i = 0; i < view.length; i += 1000) {
                        // Split operation into chunks of 1000 to not exceed call stack size
                        str += String.fromCharCode.apply(String, view.subarray(i,i+1000));
                    }
                    // Save in localstorage
                    localStorage.setItem(key+'-contents', str);
                    localStorage.setItem(key+'-length', (compressed ? 'z' : '')+length);
                    if (doneCB) doneCB(null);
                } else {
                    if (doneCB) doneCB(fr.error);
                }
            };
            fr.readAsArrayBuffer(blob);
        };
        var write = function(key, arrbuf, compressed, doneCB) {
            if (compressed) {
                /* Already compressed, no need to do anything */
                if (arrbuf instanceof ArrayBuffer) {
                    writeRaw(key, new Blob([arrbuf]), true, doneCB);
                } else {
                    // Assume it's a blob
                    writeRaw(key, arrbuf, true, doneCB);
                }
            } else {
                /* Write after compression is complete */
                var writeCompressed = function() {
                    var gzipped = compress(arrbuf);
                    writeRaw(key, new Blob([gzipped]), true, doneCB);
                };
                /* We need to compress this, to do that, it needs to be an ArrayBuffer */
                if (!(arrbuf instanceof ArrayBuffer)) {
                    var fr = new FileReader();
                    fr.onloadend = function() {
                        if (!fr.error) {
                            arrbuf = fr.result;
                            writeCompressed();
                        } else {
                            doneCB(fr.error);
                        }
                    };
                    fr.readAsArrayBuffer(arrbuf);
                } else {
                    writeCompressed();
                }
            }
        };

        /* Read file contents from localStorage */
        var read = function(key) {
            var str = localStorage.getItem(key+'-contents');
            var length = localStorage.getItem(key+'-length');
            var compressed = false;
            if (_.str.startsWith(length, 'z')) {
                compressed = true;
                length = parseInt(length.substr(1));
            } else {
                length = parseInt(length);
            }
            /* Read all the bytes into a buffer */
            var buf = new ArrayBuffer(str.length*2);
            var bufView = new Uint16Array(buf);
            for ( var i = 0; i < str.length; i++) { bufView[i] = str.charCodeAt(i) }
            /* Create a new blob from the read data, skipping possible padding byte at end */
            return {
                 data: new Blob([buf.slice(0,length)])
                ,compressed: compressed
            };
        };

        /* Check localStorage for stored files */
        var getFileList = function() {
            var list = [];
            var keys = _.keys(localStorage);
            var contents = _.map(_.filter(keys, function(key) { return _.str.endsWith(key,'-contents') }), function(key) { return key.slice(0,-9) });
            var lengths = _.map(_.filter(keys, function(key) { return _.str.endsWith(key,'-length') }), function(key) { return key.slice(0,-7) });
            return _.intersection(contents,lengths);
        };
        var hasFile = function(key) {
            var keys = _.keys(localStorage);
            return (_.contains(keys, key+'-contents') && _.contains(keys, key+'-length'));
        };

        /* Remove files from localStorage */
        var remove = function(key) {
            localStorage.removeItem(key+'-contents');
            localStorage.removeItem(key+'-length');
        };
        var removeAll = function() {
            _.each(getFileList(), remove);
        };

        /* Expose functions to outside */
        return {
             write: write
            ,writeRaw: writeRaw
            ,read: read
            ,remove: remove
            ,removeAll: removeAll
            ,getFileList: getFileList
            ,hasFile: hasFile
    }})

    /* Read parsed output files from simulator */
    .factory('parseSimulatorOutputData', function() {
        var decodeUtf8 = function(arrayBuffer, from, to) {
            var result = "";
            var i = from;
            var c = 0;
            var c1 = 0;
            var c2 = 0;

            var data = new Uint8Array(arrayBuffer);

            // If we have a BOM skip it
            if (data.length >= i+3 && data[i] === 0xef && data[i+1] === 0xbb && data[i+2] === 0xbf) {
                i += 3;
            }

            while (i < to) {
                c = data[i];

                if (c < 128) {
                    result += String.fromCharCode(c);
                    i++;
                } else if (c > 191 && c < 224) {
                    if( i+1 >= data.length ) {
                        throw "UTF-8 Decode failed. Two byte character was truncated.";
                    }
                    c2 = data[i+1];
                    result += String.fromCharCode( ((c&31)<<6) | (c2&63) );
                    i += 2;
                } else {
                    if (i+2 >= data.length) {
                        throw "UTF-8 Decode failed. Multi byte character was truncated.";
                    }
                    c2 = data[i+1];
                    c3 = data[i+2];
                    result += String.fromCharCode( ((c&15)<<12) | ((c2&63)<<6) | (c3&63) );
                    i += 3;
                }
            }
            return result;
        }

        var parseData = function(data, cb) {
            var buf;
            var handleData = function() {
                try {
                    var dv = new DataView(buf);
                    // Parse given data
                    var index = dv.getUint16(0);
                    var numLength = dv.getUint32(2);
                    var tagLength = dv.getUint16(6);
                    var tag = decodeUtf8(buf, 8, 8+tagLength);
                    var nums = [];
                    var numOffset = 8+tagLength;
                    // Read all numbers in loop, also calculate min and max at same time
                    var max = -Infinity, min = Infinity;
                    for (var i = 0; i < numLength; ++i) {
                        var n = dv.getFloat32(numOffset+4*i);
                        max = Math.max(max,n);
                        min = Math.min(min,n);
                        nums.push(n);
                    }
                    // Assemble output
                    var output = {
                        tag: tag,
                        index: index,
                        max: max,
                        min: min,
                        data: nums
                    };
                    cb(null, output);
                } catch (error) {
                    cb(error);
                }
            };
            if (data instanceof Blob) {
                var fr = new FileReader();
                fr.onloadend = function() {
                    if (fr.error) cb(fr.error);
                    else {
                        buf = fr.result;
                        handleData();
                    }
                };
                fr.readAsArrayBuffer(data);
            } else {
                buf = data;
                handleData();
            }
        };

        // Expose parser to outside
        return parseData;
    })
})();
