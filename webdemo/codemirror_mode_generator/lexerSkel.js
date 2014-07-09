var lexer = ( function() {
    var r = [
        ##regexes##
    ];
    var a = [
        ##actions##
    ];
    /* The actual lexing function */
    return function(stream, lineno) {
        /* Find all matches */
        var matches = [], lengths = [];
        for (var i = 0; i < r.length; i++) {
            var m = stream.match(r[i],false);
            if (m) {
                matches.push(i);
                lengths.push(m[0].length);
            }
        }
        /* Select the first longest match */
        var max = -1, maxI;
        for (var i = 0; i < matches.length; i++) {
            if (lengths[i] > max) {
                max = lengths[i];
                maxI = matches[i];
            }
        }
        /* If match found, use that */
        if (maxI !== undefined) {
            var m = stream.match(r[maxI]);
            return (a[maxI])(m[0],lineno);
        }
        /* None found, return nothing, this shouldn't really happen */
        return undefined;
    };
})();
