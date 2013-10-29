/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"
#include "ASTNodes.hpp"
#include "ParseActions.hpp"
#include <sstream>



// ------ Parsing event handlers ------


SequenceNode* handleProgram(SequenceNode* lineblocknode)
{
    SymbolTable::setProgram(lineblocknode);
    return lineblocknode;
}



Node* handleNumber(const double num)
{
    return new NumberNode(num);
}



Node* handleIdentifier(const std::string& name)
{
    if (SymbolTable::isVariableDeclared(name)) {
        return new VarNode(name);
    } else {
        if (SymbolTable::isFunctionDeclared(name)) {
            return new FuncRefNode(name);
        } else {
            // This is a small problem: we want the error below, to catch
            // usage of undeclared identifiers, but the function start section
            // would then generate errors, because we are not yet in the function
            // scope.

            // std::string e("unknown identifier ");
            // e += name;
            // yyerror(e.c_str());
            return new JustAnIdentifierNode(name);
        }
    }
}



VarDeclNode* handleDeclaration(const std::string& name, TypeNode* type)
{
    EquelleType t = type->type();
    SymbolTable::declareVariable(name, t);
    return new VarDeclNode(name, type);
}



VarAssignNode* handleAssignment(const std::string& name, Node* expr)
{
    // If already declared...
    if (SymbolTable::isVariableDeclared(name)) {
        // Check if already assigned.
        if (SymbolTable::isVariableAssigned(name) && !SymbolTable::variableType(name).isMutable()) {
            std::string err_msg = "variable already assigned, cannot re-assign ";
            err_msg += name;
            yyerror(err_msg.c_str());
            return nullptr;
        }
        // Check that declared type matches right hand side.
        EquelleType lhs_type = SymbolTable::variableType(name);
        EquelleType rhs_type = expr->type();
        if (lhs_type != rhs_type) {
            // Check for special case: variable declared to have type
            // 'Collection Of <Entity> Subset Of <Some entityset>',
            // actual setting its grid mapping is postponed until the entityset
            // has been created by a function call.
            // That means we have to set the actual grid mapping here.
            if (lhs_type.gridMapping() == PostponedDefinition
                && lhs_type.basicType() == rhs_type.basicType()
                && lhs_type.isCollection() && rhs_type.isCollection()
                && SymbolTable::isSubset(rhs_type.gridMapping(), lhs_type.subsetOf())) {
                // OK, should make postponed definition of the variable.
                SymbolTable::setVariableType(name, rhs_type);
                SymbolTable::setEntitySetName(rhs_type.gridMapping(), name);
            } else {
                std::string err_msg = "mismatch between type in assignment and declaration for ";
                err_msg += name;
                yyerror(err_msg.c_str());
                return nullptr;
            }
        }
    } else {
        // Setting the gridmapping, as in the block above. Only this is
        // the direct (no previous declaration) assignment scenario.
        const int gm = expr->type().gridMapping();
        if (gm != NotApplicable && SymbolTable::entitySetName(gm) == "AnonymousEntitySet") {
            SymbolTable::setEntitySetName(gm, name);
        }
        SymbolTable::declareVariable(name, expr->type());
    }

    // Set variable to assigned (unless mutable) and return.
    if (!SymbolTable::variableType(name).isMutable()) {
        SymbolTable::setVariableAssigned(name, true);
    }
    return new VarAssignNode(name, expr);
}



Node* handleFuncDeclaration(const std::string& name, FuncTypeNode* ftype)
{
    SymbolTable::renameCurrentFunction(name);
    SymbolTable::retypeCurrentFunction(ftype->funcType());
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    return new FuncDeclNode(name, ftype);
}



Node* handleFuncStart(const std::string& name, Node* funcargs)
{
    SymbolTable::setCurrentFunction(name);
    return new FuncStartNode(name, funcargs);
}



void handleFuncStartType()
{
    SymbolTable::declareFunction("TemporaryFunction");
    SymbolTable::setCurrentFunction("TemporaryFunction");
}



SequenceNode* handleBlock(SequenceNode* block)
{
    // This is called after the block AST has been constructed,
    // so we should switch back to Main scope.
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    return block;
}



FuncAssignNode* handleFuncAssignment(Node* funcstart, SequenceNode* fbody)
{
    return new FuncAssignNode(funcstart, fbody);
}



ReturnStatementNode* handleReturnStatement(Node* expr)
{
    return new ReturnStatementNode(expr);
}



Node* handleDeclarationAssign(const std::string& name, TypeNode* type, Node* expr)
{
    SequenceNode* seq = new SequenceNode;
    seq->pushNode(handleDeclaration(name, type));
    seq->pushNode(handleAssignment(name, expr));
    return seq;
}



TypeNode* handleCollection(TypeNode* btype, Node* gridmapping, Node* subsetof)
{
    assert(gridmapping == nullptr || subsetof == nullptr);
    EquelleType bt = btype->type();
    if (!bt.isBasic()) {
        std::string errmsg = "attempting to declare a Collection Of <nonsense>";
        yyerror(errmsg.c_str());
    }
    int gm = NotApplicable;
    if (gridmapping) {
        if (!gridmapping->type().isEntityCollection() || gridmapping->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be On a Collection of Cell, Face etc.");
        } else {
            gm = gridmapping->type().gridMapping();
        }
    }
    int subset = NotApplicable;
    if (subsetof) {
        // We are creating a new entity collection.
        if (!subsetof->type().isEntityCollection() || subsetof->type().gridMapping() == NotApplicable) {
            yyerror("a Collection must be Subset Of a Collection of Cell, Face etc.");
        } else {
            gm = PostponedDefinition;
            subset = subsetof->type().gridMapping();
        }
    }
    delete btype;
    delete gridmapping;
    delete subsetof;
    return new TypeNode(EquelleType(bt.basicType(), Collection, gm, subset));
}



FuncTypeNode* handleFuncType(FuncArgsDeclNode* argtypes, TypeNode* rtype)
{
    return new FuncTypeNode(FunctionType(argtypes->arguments(), rtype->type()));
}



FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args)
{
    const Function& f = SymbolTable::getFunction(name);
    // Check function call arguments.
    const auto argtypes = args->argumentTypes();
    if (argtypes.size() != f.functionType().arguments().size()) {
        std::string err_msg = "wrong number of arguments when calling function ";
        err_msg += name;
        yyerror(err_msg.c_str());
    }
    // At the moment, we do not check function argument types.
    // If the function returns a new entity set, we must declare it (even if anonymous).
    int dynsubret = f.functionType().dynamicSubsetReturn(args->argumentTypes());
    if (dynsubret != NotApplicable) {
        // Create a new entity collection.
        const int gm = SymbolTable::declareNewEntitySet("AnonymousEntitySet", dynsubret);
        return new FuncCallNode(name, args, gm);
    } else {
        return new FuncCallNode(name, args);
    }
}



FuncCallStatementNode* handleFuncCallStatement(FuncCallNode* fcall)
{
    return new FuncCallStatementNode(fcall);
}



BinaryOpNode* handleBinaryOp(BinaryOp op, Node* left, Node* right)
{
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if (!isNumericType(lt.basicType()) || !(isNumericType(rt.basicType()))) {
        yyerror("arithmetic binary operators only apply to numeric types");
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            yyerror("arithmetic binary operators on Collections only acceptable "
                    "if both sides are On the same set.");
        }
    }
    switch (op) {
    case Add:
        // Intentional fall-through.
    case Subtract:
        if (lt != rt) {
            yyerror("addition and subtraction only allowed between identical types.");
        }
        break;
    case Multiply:
        if (lt.basicType() == Vector && rt.basicType() == Vector) {
            yyerror("cannot multiply two 'Vector' types.");
        }
        break;
    case Divide:
        if (rt.basicType() != Scalar) {
            yyerror("can only divide by 'Scalar' types");
        }
        break;
    default:
        yyerror("internal compiler error in handleBinaryOp().");
    }
    return new BinaryOpNode(op, left, right);
}



ComparisonOpNode* handleComparison(ComparisonOp op, Node* left, Node* right)
{
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if ((lt.basicType() != Scalar) || (rt.basicType() != Scalar)) {
        yyerror("comparison operators can only be applied to scalars");
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            yyerror("comparison operators on Collections only acceptable "
                    "if both sides are On the same set.");
        }
    }
    return new ComparisonOpNode(op, left, right);
}



NormNode* handleNorm(Node* expr_to_norm)
{
    const BasicType bt = expr_to_norm->type().basicType();
    if (isEntityType(bt) || bt == Scalar || bt == Vector) {
        return new NormNode(expr_to_norm);
    } else {
        yyerror("can only take norm of Scalar, Vector, Cell, Face, Edge and Vertex types.");
        return new NormNode(expr_to_norm);
    }
}



UnaryNegationNode* handleUnaryNegation(Node* expr_to_negate)
{
    if (!isNumericType(expr_to_negate->type().basicType())) {
        yyerror("unary minus can only be applied to numeric types.");
    }
    return new UnaryNegationNode(expr_to_negate);
}



TrinaryIfNode* handleTrinaryIf(Node* predicate, Node* iftrue, Node* iffalse)
{
    const EquelleType pt = predicate->type();
    const EquelleType tt = iftrue->type();
    const EquelleType ft = iffalse->type();
    if (pt.basicType() != Bool) {
        yyerror("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<predicate> must be a Bool type.");
    }
    if (tt != ft) {
        yyerror("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<iftrue> and <iffalse> must have the same type.");
    }
    if ((pt.isCollection() != tt.isCollection()) ||
        (pt.gridMapping() != tt.gridMapping())) {
        yyerror("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "all three expressions must be 'On' the same set.");
    }
    return new TrinaryIfNode(predicate, iftrue, iffalse);
}



OnNode* handleOn(Node* left, Node* right)
{
    if (!right->type().isEntityCollection()) {
        yyerror("in a '<left> On <right>' expression "
                "the expression <right> must be a Collection Of Cell, Face, Edge or Vertex.");
    }
    if (left->type().isSequence()) {
        yyerror("cannot use On operator with a Sequence.");
    }

    if (left->type().isCollection()) {
        const int gml = left->type().gridMapping();
        if (SymbolTable::entitySetType(gml) != right->type().basicType()) {
            std::string err_msg;
            err_msg += "in a '<left> On <right>' expression the expression <right> must "
                "be a collection of the same kind of that which <left> is On. ";
            err_msg += "Collection on the left is On ";
            err_msg += SymbolTable::entitySetName(gml);
            yyerror(err_msg.c_str());
        }
        // Following test is wrong: cannot deal properly with rhs that is not an entity set.
        // const int gmr = right->type().gridMapping();
        // if (!SymbolTable::isSubset(gml, gmr)
        //     && !SymbolTable::isSubset(gmr, gml)) {
        //     yyerror("in a '<left> On <right>' expression "
        //             "the entityset <right> must be a (non-strict) super- or sub-set of "
        //             "the set which <left> is 'On'.");
        // }
    }
    return new OnNode(left, right);
}



StringNode* handleString(const std::string& content)
{
    return new StringNode(content);
}



TypeNode* handleMutableType(TypeNode* type_expr)
{
    EquelleType et = type_expr->type();
    et.setMutable(true);
    TypeNode* tn = new TypeNode(et);
    delete type_expr;
    return tn;
}



TypeNode* handleSequence(TypeNode* basic_type)
{
    const EquelleType et = basic_type->type();
    if (!et.isBasic()) {
        yyerror("cannot create a Sequence of non-basic types.");
    }
    return new TypeNode(EquelleType(et.basicType(), Sequence, et.gridMapping(), et.subsetOf()));
}



LoopNode* handleLoopStart(const std::string& loop_variable, const std::string& loop_set)
{
    // Check that loop_set is a sequence, extract its type.
    EquelleType loop_set_type;
    if (SymbolTable::isVariableDeclared(loop_set)) {
        loop_set_type = SymbolTable::variableType(loop_set);
        if (!loop_set_type.isSequence()) {
            std::string err_msg = "loop set must be a Sequence: ";
            err_msg += loop_set;
            yyerror(err_msg.c_str());
        }
    } else {
        std::string err_msg = "unknown variable used: ";
        err_msg += loop_set;
        yyerror(err_msg.c_str());
    }
    // Create LoopNode
    LoopNode* ln = new LoopNode(loop_variable, loop_set);
    // Create a name for the loop scope.
    static int next_loop_index = 0;
    std::ostringstream os;
    os << "ForLoopWithIndex" << next_loop_index++;
    // Set name in loop node, declare scope and
    // set to current.
    ln->setName(os.str());
    SymbolTable::declareFunction(os.str());
    SymbolTable::setCurrentFunction(os.str());
    // Declare loop variable
    SymbolTable::declareVariable(loop_variable, loop_set_type.basicType());
    return ln;
}



LoopNode* handleLoopStatement(LoopNode* loop_start, SequenceNode* loop_block)
{
    loop_start->setBlock(loop_block);
    return loop_start;
}

