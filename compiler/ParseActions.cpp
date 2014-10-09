/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#include "Common.hpp"
#include "EquelleType.hpp"
#include "ASTNodes.hpp"
#include "ParseActions.hpp"
#include "Dimension.hpp"
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <cmath>



// ------ Parsing event handlers ------


SequenceNode* handleProgram(SequenceNode* lineblocknode)
{
    SymbolTable::setProgram(lineblocknode);
    return lineblocknode;
}



NumberNode* handleNumber(const double num)
{
    return new NumberNode(num);
}



QuantityNode* handleQuantity(NumberNode* number, UnitNode* unit)
{
    return new QuantityNode(number, unit);
}



UnitNode* handleUnit(const std::string& name)
{
    return new UnitNode(name);
}



UnitNode* handleUnitOp(BinaryOp op, UnitNode* left, UnitNode* right)
{
    Dimension d = left->dimension();
    double c = left->conversionFactorSI();
    switch (op) {
    case Multiply:
        d = d + right->dimension();
        c = c * right->conversionFactorSI();
        break;
    case Divide:
        d = d - right->dimension();
        c = c / right->conversionFactorSI();
        break;
    default:
        yyerror("Units can only be manipulated with '*', '/' or '^'.");
    }
    delete left;
    delete right;
    return new UnitNode(d, c);
}



UnitNode* handleUnitPower(UnitNode* unit, const double num)
{
    const int n = static_cast<int>(num);
    if (n != num) {
        yyerror("Powers of units (to the right of '^') can only be integers.");
    }
    const Dimension d = unit->dimension() * n;
    const double c = std::pow(unit->conversionFactorSI(), n);
    delete unit;
    return new UnitNode(d, c);
}



ExpressionNode* handleIdentifier(const std::string& name)
{
    return new JustAnIdentifierNode(name);
#if 0
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
#endif
}



VarDeclNode* handleDeclaration(const std::string& name, TypeNode* type)
{
    return new VarDeclNode(name, type);
#if 0
    EquelleType t = type->type();
    SymbolTable::declareVariable(name, t);
    return new VarDeclNode(name, type);
#endif
}



VarAssignNode* handleAssignment(const std::string& name, ExpressionNode* expr)
{
    return new VarAssignNode(name, expr);
#if 0
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
                && lhs_type.isStencil() == rhs_type.isStencil()
                && rhs_type.isDomain()
                && SymbolTable::isSubset(rhs_type.gridMapping(), lhs_type.subsetOf())) {
                // OK, should make postponed definition of the variable.
                SymbolTable::setVariableType(name, rhs_type);
                SymbolTable::setVariableDimension(name, expr->dimension());
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
        SymbolTable::setVariableDimension(name, expr->dimension());
    }

    // Set variable to assigned (unless mutable) and return.
    if (!SymbolTable::variableType(name).isMutable()) {
        SymbolTable::setVariableAssigned(name, true);
    }
    return new VarAssignNode(name, expr);
#endif
}



Node* handleFuncDeclaration(const std::string& name, FuncTypeNode* ftype)
{
#if 0
    return new FuncDeclNode(name, ftype);
#endif
    SymbolTable::renameCurrentFunction(name);
    SymbolTable::retypeCurrentFunction(ftype->funcType());
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    return new FuncDeclNode(name, ftype);
}



void handleFuncStartType()
{
#if 0
    SymbolTable::declareFunction("TemporaryFunction");
    SymbolTable::setCurrentFunction("TemporaryFunction");
#endif
}


FuncCallLikeNode* handleFuncAssignmentStart(const std::string& name, FuncArgsNode* args)
{
    return new FuncStartNode(name, args);
#if 0
    // We are dealing with a function
    if (SymbolTable::isFunctionDeclared(name)) {
        // Set the scope name for the following block (the function itself)
        // Will be "undone" in handleFuncAssignment
        SymbolTable::setCurrentFunction(name);
        return new FuncStartNode(name, args);
    }
    // We are dealing with a defined stencil variable
    else if (SymbolTable::isVariableDeclared(name)) {
        return handleStencilAccess(name, args);
    }
    // We are dealing with an undefined stencil variable
    else {
        EquelleType type(Invalid, Collection);
        type.setStencil(true);
        SymbolTable::declareVariable(name, type);
        return handleStencilAccess(name, args);
    }
#endif
}


FuncAssignNode* handleFuncAssignment(Node* funcstart, SequenceNode* fbody)
{
    return new FuncAssignNode(funcstart, fbody);
#if 0
    // This is called after the block AST has been constructed,
    // so we should switch back to Main scope.
    // See also handleFuncAssignmentStart
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    return new FuncAssignNode(funcstart, fbody);
#endif
}



ReturnStatementNode* handleReturnStatement(ExpressionNode* expr)
{
    return new ReturnStatementNode(expr);
}



Node* handleDeclarationAssign(const std::string& name, TypeNode* type, ExpressionNode* expr)
{
    SequenceNode* seq = new SequenceNode;
    seq->pushNode(handleDeclaration(name, type));
    seq->pushNode(handleAssignment(name, expr));
    return seq;
}



CollectionTypeNode* handleCollection(TypeNode* btype, ExpressionNode* gridmapping, ExpressionNode* subsetof)
{
    return new CollectionTypeNode(btype, gridmapping, subsetof);
#if 0
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
#endif
}

TypeNode* handleStencilCollection(TypeNode* type_expr)
{
    EquelleType et = type_expr->type();
    et.setStencil(true);
    TypeNode* tn = new TypeNode(et);
    delete type_expr;
    return tn;
}


FuncTypeNode* handleFuncType(FuncArgsDeclNode* argtypes, TypeNode* rtype)
{
    return new FuncTypeNode(FunctionType(argtypes->arguments(), rtype->type()));
}



StencilNode* handleStencilAccess(const std::string& name, FuncArgsNode* args)
{
    return new StencilNode(name, args);
#if 0
    if (!SymbolTable::isVariableDeclared(name)) {
        std::string err_msg = "Could not find the stencil variable " + name;
        yyerror(err_msg.c_str());
    }

    auto argtypes = args->argumentTypes();
    for (int i=0; i<argtypes.size(); ++i) {
        if (!isStencilType(argtypes[i].basicType())) {
            std::stringstream err_msg;
            err_msg << "Cannot access a stencil with a non-stencil index in variable \""
                    << name << "\"" << std::endl;
            yyerror(err_msg.str().c_str());
        }
        else if (argtypes[i].basicType() != StencilI + i) {
            std::stringstream err_msg;
            err_msg << "Got index " << basicTypeString(argtypes[i].basicType())
                    << " but expected " << basicTypeString(BasicType(StencilI + i))
                    << " for variable \"" << name << "\"" << std::endl;
            yyerror(err_msg.str().c_str());
        }
    }
    return new StencilNode(name, args);
#endif
}

FuncCallNode* handleFuncCall(const std::string& name, FuncArgsNode* args)
{
    return new FuncCallNode(name, args);
#if 0
    const Function& f = SymbolTable::getFunction(name);
    // Check function call arguments.
    const auto argtypes = args->argumentTypes();
    if (argtypes.size() != f.functionType().arguments().size()) {
        std::string err_msg = "wrong number of arguments when calling function ";
        err_msg += name;
        yyerror(err_msg.c_str());
    }
    // At the moment, we do not check function argument types.
    // If the function returns a new dynamically created domain,
    // we must declare it (anonymously for now).
    const EquelleType rtype = f.returnType(argtypes);
    if (rtype.isDomain()) {
        const int dynsubret = f.functionType().dynamicSubsetReturn(argtypes);
        if (dynsubret != NotApplicable) {
            // Create a new domain.
            const int gm = SymbolTable::declareNewEntitySet("AnonymousEntitySet", dynsubret);
            return new FuncCallNode(name, args, gm);
        }
    }
    return new FuncCallNode(name, args);
#endif
}

FuncCallLikeNode* handleFuncCallLike(const std::string& name, FuncArgsNode* args)
{
    return handleFuncCall(name, args);
#if 0
    if (SymbolTable::isFunctionDeclared(name)) {
        return handleFuncCall(name, args);
    }
    else if (SymbolTable::isVariableDeclared(name)) {
        return handleStencilAccess(name, args);
    }
    else {
        std::string err_msg = "Could not find the stencil variable or function " + name;
        yyerror(err_msg.c_str());
        return nullptr;
    }
#endif
}

FuncCallStatementNode* handleFuncCallStatement(FuncCallLikeNode* fcall_like)
{
    FuncCallNode* fcall = dynamic_cast<FuncCallNode*>(fcall_like);
    if (fcall == nullptr) {
        std::string err_msg = "Internal error: The function \"" + fcall_like->name() + "\" does not appear to be properly defined";
        yyerror(err_msg.c_str());
    }
    return new FuncCallStatementNode(fcall);
}



BinaryOpNode* handleBinaryOp(BinaryOp op, ExpressionNode* left, ExpressionNode* right)
{
    return new BinaryOpNode(op, left, right);
#if 0
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if (!isNumericType(lt.basicType()) || !(isNumericType(rt.basicType()))) {
        yyerror("arithmetic binary operators only apply to numeric types");
    }
    if (lt.isArray() || rt.isArray()) {
        yyerror("arithmetic binary operators cannot be applied to Array types");
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
            if ((lt.basicType() == StencilI || lt.basicType() == StencilJ || lt.basicType() == StencilK)
                && rt.basicType() == Scalar) {
                //i,j,k OP n is OK
            }
            else if (lt.basicType() == Scalar &&
                     (rt.basicType() == StencilI || rt.basicType() == StencilJ || rt.basicType() == StencilK)) {
                //n OP i,j,k is OK
            }
            else if ((lt.isStencil() && rt.basicType() == Scalar)
                     || (rt.isStencil() && lt.basicType() == Scalar)) {
                //n OP u(i, j)  and  u(i, j) OP n is OK
            }
            else {
                yyerror("addition and subtraction only allowed between identical types.");
            }
        }
        if (left->dimension() != right->dimension()) {
            yyerror("addition and subtraction only allowed when both sides have same dimension.");
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
#endif
}



ComparisonOpNode* handleComparison(ComparisonOp op, ExpressionNode* left, ExpressionNode* right)
{
    return new ComparisonOpNode(op, left, right);
#if 0
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if ((lt.basicType() != Scalar) || (rt.basicType() != Scalar)) {
        yyerror("comparison operators can only be applied to scalars");
    }
    if (lt.isArray() || rt.isArray()) {
        yyerror("comparison operators cannot be applied to Array types");
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            yyerror("comparison operators on Collections only acceptable "
                    "if both sides are On the same set.");
        }
    }
    if (left->dimension() != right->dimension()) {
        yyerror("comparison operators only allowed when both sides have same dimension.");
    }
    return new ComparisonOpNode(op, left, right);
#endif
}



NormNode* handleNorm(ExpressionNode* expr_to_norm)
{
    return new NormNode(expr_to_norm);
#if 0
    if (expr_to_norm->type().isArray()) {
        yyerror("cannot take norm of an Array.");
    }
    const BasicType bt = expr_to_norm->type().basicType();
    if (isEntityType(bt) || bt == Scalar || bt == Vector) {
        return new NormNode(expr_to_norm);
    } else {
        yyerror("can only take norm of Scalar, Vector, Cell, Face, Edge and Vertex types.");
        return new NormNode(expr_to_norm);
    }
#endif
}



UnaryNegationNode* handleUnaryNegation(ExpressionNode* expr_to_negate)
{
    return new UnaryNegationNode(expr_to_negate);
#if 0
    if (!isNumericType(expr_to_negate->type().basicType())) {
        yyerror("unary minus can only be applied to numeric types.");
    }
    if (expr_to_negate->type().isArray()) {
        yyerror("unary minus cannot be applied to an Array.");
    }
    return new UnaryNegationNode(expr_to_negate);
#endif
}



TrinaryIfNode* handleTrinaryIf(ExpressionNode* predicate, ExpressionNode* iftrue, ExpressionNode* iffalse)
{
    return new TrinaryIfNode(predicate, iftrue, iffalse);
#if 0
    const EquelleType pt = predicate->type();
    const EquelleType tt = iftrue->type();
    const EquelleType ft = iffalse->type();
    if (pt.isArray() || tt.isArray() || ft.isArray()) {
        yyerror("in trinary if operator, no operands can be of Array type.");
    }
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
    if (iftrue->dimension() != iffalse->dimension()) {
        yyerror("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<iftrue> and <iffalse> must have the same dimension.");
    }
    return new TrinaryIfNode(predicate, iftrue, iffalse);
#endif
}



OnNode* handleOn(ExpressionNode* left, ExpressionNode* right)
{
    return new OnNode(left, right, false);
#if 0
    const EquelleType lt = left->type();
    const EquelleType rt = right->type();
    if (lt.isArray() || rt.isArray()) {
        yyerror("cannot use On operator with an Array.");
    }
    // Left side can be anything but a sequence.
    if (lt.isSequence()) {
        yyerror("cannot use On operator with a Sequence.");
    }
    // Right side must be an entity collection.
    if (!rt.isEntityCollection()) {
        yyerror("in a '<left> On <right>' expression "
                "the expression <right> must be a Collection Of Cell, Face, Edge or Vertex.");
    }
    // Left side must be some collection.
    if (!lt.isCollection()) {
        yyerror("in a '<left> On <right>' expression "
                "the expression <left> must be a Collection.");
    } else {
        // The domain (grid mapping) of the left side must contain
        // the right hand side. Explanation by example:
        //   x = AllCells()
        //   y : Collection Of Scalar On x = |x|
        // The above defines x and y, and the On part declares that there is a 1-1 mapping x -> y.
        // We can even denote this mapping y(x).
        //   z = InteriorFaces()
        //   w : Collection Of Cell On z = FirstCell(z)
        // Now w is On z, meaning that there is a 1-1 mapping z -> w, we denote it w(z)
        // Finally, what does then the following mean?
        //   yow = y On w
        // Its meaning is intended to be a composition of mappings, that is
        // there is now a 1-1 mapping z->yow defined by yow(z) = y(w(z)).
        // This is only ok if the range of w(z) is contained in the domain of y(x), that is x.
        // In our case that means that the entity collection on the right hand side must be contained
        // in the domain of the left.
        const int left_domain = lt.gridMapping();
        const int right_collection = rt.subsetOf();
        if (!SymbolTable::isSubset(right_collection, left_domain)) {
            std::string err_msg;
            err_msg += "in a '<left> On <right>' expression the expression <right> must "
                "be a collection that is contained in the domain that <left> is On. ";
            err_msg += "Collection on the left is On ";
            err_msg += SymbolTable::entitySetName(left_domain);
            yyerror(err_msg.c_str());
        }
    }
    return new OnNode(left, right, false);
#endif
}



OnNode* handleExtend(ExpressionNode* left, ExpressionNode* right)
{
    return new OnNode(left, right, true);
#if 0
    const EquelleType lt = left->type();
    const EquelleType rt = right->type();
    if (lt.isArray() || rt.isArray()) {
        yyerror("cannot use Extend operator with an Array.");
    }
    // Left side can be anything but a sequence.
    if (lt.isSequence()) {
        yyerror("cannot use Extend operator with a Sequence.");
    }
    // Right side must be a domain.
    if (!rt.isDomain()) {
        yyerror("in a '<left> Extend <right>' expression "
                "the expression <right> must be a Collection Of Cell, Face, Edge or Vertex, "
                "that also is a domain (all unique, non-Empty elements).");
    }
    // If left side is a collection, its domain (grid mapping) must be
    // a subset of the right hand side.
    if (lt.isCollection()) {
        const int left_domain = lt.gridMapping();
        const int right_domain = lt.gridMapping();
        if (!SymbolTable::isSubset(left_domain, right_domain)) {
            std::string err_msg;
            err_msg += "in a '<left> Extend <right>' expression the expression <right> must "
                "be a domain that contains the domain that <left> is On. ";
            err_msg += "Collection on the left is On ";
            err_msg += SymbolTable::entitySetName(left_domain);
            err_msg += " and Domain on the right is On ";
            err_msg += SymbolTable::entitySetName(right_domain);
            yyerror(err_msg.c_str());
        }
    }
    return new OnNode(left, right, true);
#endif
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



TypeNode* handleArrayType(const int array_size, TypeNode* type_expr)
{
    EquelleType et = type_expr->type();
    if (et.isArray()) {
        yyerror("cannot create an Array of an Array.");
        return type_expr;
    } else {
        et.setArraySize(array_size);
        TypeNode* tn = new TypeNode(et);
        delete type_expr;
        return tn;
    }
}



ArrayNode* handleArray(FuncArgsNode* expr_list)
{
    return new ArrayNode(expr_list);
#if 0
    const auto& elems = expr_list->arguments();
    if (elems.empty()) {
        yyerror("cannot create an empty array.");
    } else {
        const EquelleType et = elems[0]->type();
        if (et.isArray()) {
            yyerror("an Array cannot contain another Array.");
        }
        for (const auto& elem : elems) {
            if (elem->type() != et) {
                yyerror("elements of an Array must all have the same type");
            }
        }
    }
    return new ArrayNode(expr_list);
#endif
}



LoopNode* handleLoopStart(const std::string& loop_variable, const std::string& loop_set)
{
    return new LoopNode(loop_variable, loop_set);
#if 0
    // Check that loop_set is a sequence, extract its type.
    EquelleType loop_set_type;
    if (SymbolTable::isVariableDeclared(loop_set)) {
        loop_set_type = SymbolTable::variableType(loop_set);
        if (!loop_set_type.isSequence()) {
            std::string err_msg = "loop set must be a Sequence: ";
            err_msg += loop_set;
            yyerror(err_msg.c_str());
        }
        if (loop_set_type.isArray()) {
            std::string err_msg = "loop set cannot be an Array: ";
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
#endif
}



LoopNode* handleLoopStatement(LoopNode* loop_start, SequenceNode* loop_block)
{
    loop_start->setBlock(loop_block);
    return loop_start;
#if 0
    loop_start->setBlock(loop_block);
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    return loop_start;
#endif
}



RandomAccessNode* handleRandomAccess(ExpressionNode* expr, const int index)
{
    return new RandomAccessNode(expr, index);
#if 0
    if (expr->type().isArray()) {
        if (index < 0 || index >= expr->type().arraySize()) {
            yyerror("index out of array bounds in '[<index>]' random access operator.");
        }
    } else if (expr->type().basicType() == Vector) {
        if (index < 0 || index > 2) {
            yyerror("cannot use '[<index>]' random access operator on a Vector with index < 0 or > 2");
        }
    } else {
        yyerror("cannot use '[<index>]' random access operator with anything other than a Vector or Array");
    }
    return new RandomAccessNode(expr, index);
#endif
}

SequenceNode* handleStencilAssignment(FuncCallLikeNode* lhs, ExpressionNode* rhs)
{
    SequenceNode* retval = new SequenceNode();

    StencilNode* stencil = dynamic_cast<StencilNode*>(lhs);
    if (stencil == nullptr) {
        std::string err_msg = "Internal error: The stencil \"" + lhs->name() + "\" does not appear to be properly defined";
        yyerror(err_msg.c_str());
    }

#if 0
    // If the type is a collection of invalids (no pun intended)
    // we can safely set it to the type of the rhs
    EquelleType lhs_et = SymbolTable::variableType(stencil->name());
    if (lhs_et.basicType() == Invalid
        && lhs_et.compositeType() == Collection
        && lhs_et.isStencil()) {
        EquelleType lhs_et = rhs->type();
        lhs_et.setMutable(true);
        SymbolTable::setVariableType(stencil->name(), lhs_et);
        // TODO: set dimensions correctly here
        TypeNode* lhs_type = new TypeNode(lhs_et);
        retval->pushNode(new VarDeclNode(stencil->name(), lhs_type));
    }
#endif
    retval->pushNode(new StencilAssignmentNode(stencil, rhs));
    return retval;
}

