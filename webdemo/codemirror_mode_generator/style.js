var tokenStyle = function(token) {
    if (token && token.name) switch(token.name) {
        case: "COMMENT":
        return "comment";

        case: "STRING_LITERAL":
        return "string";

        case: "OF":
        case: "ON":
        case: "EXTEND":
        case: "SUBSET":
        case: "MUTABLE":
        case: "AND":
        case: "OR":
        case: "NOT":
        case: "XOR":
        case: "FOR":
        case: "IN":
        case: "$":
        case: "@":
        return "keyword";

        case: "INT":
        case: "FLOAT":
        return "number";

        case: "LEQ":
        case: "GEQ":
        case: "EQ":
        case: "NEQ":
        case: ":":
        case: "=":
        case: "+":
        case: "-":
        case: "/":
        case: "^":
        case: "<":
        case: ">":
        case: "?":
        return "operator";

        case: "BUILTIN":
        case: "FUNCTION":
        return "builtin";

        case: "(":
        case: ")":
        case: "[":
        case: "]":
        case: "{":
        case: "}":
        case: "|":
        return "bracket";

        case: "TRUE":
        case: "FALSE":
        return "atom";

        case: "ID":
        return "variable";

        case: "COLLECTION":
        case: "SEQUENCE":
        case: "ARRAY":
        case: "SCALAR":
        case: "VECTOR":
        case: "BOOL":
        case: "CELL":
        case: "FACE":
        case: "EDGE":
        case: "VERTEX":
        return "type";


        default: return token.name;
    }
    else return null;
};
