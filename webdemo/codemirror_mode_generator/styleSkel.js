var tokenStyle = function(token) {
    if (token && token.name) switch(token.name) {
        ##cases##

        default: return token.name;
    }
    else return null;
};
