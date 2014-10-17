/*
  Copyright 2014 SINTEF ICT, Applied Mathematics.
*/

#include "CheckASTVisitor.hpp"
#include "ASTNodes.hpp"
#include "SymbolTable.hpp"
#include <iostream>
#include <stdexcept>
#include <sstream>


CheckASTVisitor::CheckASTVisitor()
    : checking_suppression_level_(0),
      next_loop_index_(0)
{
}

CheckASTVisitor::~CheckASTVisitor()
{
}


void CheckASTVisitor::visit(SequenceNode&)
{
}

void CheckASTVisitor::midVisit(SequenceNode&)
{
}

void CheckASTVisitor::postVisit(SequenceNode&)
{
}

void CheckASTVisitor::visit(NumberNode&)
{
}

void CheckASTVisitor::visit(QuantityNode&)
{
}

void CheckASTVisitor::postVisit(QuantityNode&)
{
}

void CheckASTVisitor::postVisit(BinaryOpUnitNode& node)
{
    if (node.op() != Multiply && node.op() != Divide) {
        error("units can only be manipulated with '*', '/' or '^'.", node.location());
    }
}

void CheckASTVisitor::visit(StringNode&)
{
}

void CheckASTVisitor::visit(TypeNode&)
{
}

void CheckASTVisitor::visit(CollectionTypeNode&)
{
}

void CheckASTVisitor::postVisit(CollectionTypeNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    EquelleType bt = node.baseType()->type();
    if (!bt.isBasic()) {
        std::string errmsg = "attempting to declare a Collection Of <something other than a basic type>";
        error(errmsg, node.location());
    }
    const ExpressionNode* gridmapping = node.gridMapping();
    if (gridmapping) {
        if (!gridmapping->type().isEntityCollection() || gridmapping->type().gridMapping() == NotApplicable) {
            error("a Collection must be On a Collection of Cell, Face etc.", node.location());
        }
    }
    const ExpressionNode* subsetof = node.subsetOf();
    if (subsetof) {
        // We are creating a new entity collection.
        if (!subsetof->type().isEntityCollection() || subsetof->type().gridMapping() == NotApplicable) {
            error("a Collection must be Subset Of a Collection of Cell, Face etc.", node.location());
        }
    }
}

void CheckASTVisitor::visit(FuncTypeNode&)
{
}

void CheckASTVisitor::visit(BinaryOpNode&)
{
}

void CheckASTVisitor::midVisit(BinaryOpNode&)
{
}

void CheckASTVisitor::postVisit(BinaryOpNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const EquelleType lt = node.left()->type();
    const EquelleType rt = node.right()->type();
    const BinaryOp op = node.op();
    if (!isNumericType(lt.basicType()) || !(isNumericType(rt.basicType()))) {
        error("arithmetic binary operators only apply to numeric types", node.location());
        return;
    }
    if (lt.isArray() || rt.isArray()) {
        error("arithmetic binary operators cannot be applied to Array types", node.location());
        return;
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            error("arithmetic binary operators on Collections only acceptable "
                    "if both sides are On the same set.", node.location());
            return;
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
                error("addition and subtraction only allowed between identical types.", node.location());
                return;
            }
        }
        if (node.left()->dimension() != node.right()->dimension()) {
            error("addition and subtraction only allowed when both sides have same dimension.", node.location());
            return;
        }
        break;
    case Multiply:
        if (lt.basicType() == Vector && rt.basicType() == Vector) {
            error("cannot multiply two 'Vector' types.", node.location());
            return;
        }
        break;
    case Divide:
        if (rt.basicType() != Scalar) {
            error("can only divide by 'Scalar' types", node.location());
            return;
        }
        break;
    default:
        error("internal compiler error in CheckASTVisitor::postVisit(BinaryOpNode&).", node.location());
    }
}

void CheckASTVisitor::visit(ComparisonOpNode&)
{
}

void CheckASTVisitor::midVisit(ComparisonOpNode&)
{
}

void CheckASTVisitor::postVisit(ComparisonOpNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const ExpressionNode* left = node.left();
    const ExpressionNode* right = node.right();
    EquelleType lt = left->type();
    EquelleType rt = right->type();
    if ((lt.basicType() != Scalar) || (rt.basicType() != Scalar)) {
        error("comparison operators can only be applied to scalars", node.location());
        return;
    }
    if (lt.isArray() || rt.isArray()) {
        error("comparison operators cannot be applied to Array types", node.location());
        return;
    }
    if (lt.isCollection() && rt.isCollection()) {
        if (lt.gridMapping() != rt.gridMapping()) {
            error("comparison operators on Collections only acceptable "
                  "if both sides are On the same set.", node.location());
            return;
        }
    }
    if (left->dimension() != right->dimension()) {
        error("comparison operators only allowed when both sides have same dimension.", node.location());
        return;
    }
}

void CheckASTVisitor::visit(NormNode&)
{
}

void CheckASTVisitor::postVisit(NormNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const ExpressionNode* expr = node.normedExpression();
    if (expr->type().isArray()) {
        error("cannot take norm of an Array.", node.location());
        return;
    }
    const BasicType bt = expr->type().basicType();
    const bool ok_type = isEntityType(bt) || bt == Scalar || bt == Vector;
    if (!ok_type) {
        error("can only take norm of Scalar, Vector, Cell, Face, Edge and Vertex types.", node.location());
    }
}

void CheckASTVisitor::visit(UnaryNegationNode&)
{
}

void CheckASTVisitor::postVisit(UnaryNegationNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const ExpressionNode* expr = node.negatedExpression();
    if (!isNumericType(expr->type().basicType())) {
        error("unary minus can only be applied to numeric types.", node.location());
        return;
    }
    if (expr->type().isArray()) {
        error("unary minus cannot be applied to an Array.", node.location());
    }
}

void CheckASTVisitor::visit(OnNode&)
{
}

void CheckASTVisitor::midVisit(OnNode&)
{
}

void CheckASTVisitor::postVisit(OnNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const EquelleType lt = node.leftType();
    const EquelleType rt = node.rightType();
    // No side can be an array.
    if (lt.isArray() || rt.isArray()) {
        error("cannot use On or Extend operator with an Array.", node.location());
        return;
    }
    // Left side can be anything but a sequence.
    if (lt.isSequence()) {
        error("cannot use On or Extend operator with a Sequence.", node.location());
        return;
    }
    const bool extend = node.isExtend();
    if (extend) {
        // This is for the Extend operator.
        // Right side must be a domain.
        if (!rt.isDomain()) {
            error("in a '<left> Extend <right>' expression "
                  "the expression <right> must be a Collection Of Cell, Face, Edge or Vertex, "
                  "that also is a domain (all unique, non-Empty elements).", node.location());
            return;
        }
        // If left side is a collection, its domain (grid mapping) must be
        // a subset of the right hand side.
        if (lt.isCollection()) {
            const int left_domain = lt.gridMapping();
            const int right_domain = rt.gridMapping();
            if (!SymbolTable::isSubset(left_domain, right_domain)) {
                std::string err_msg;
                err_msg += "in a '<left> Extend <right>' expression the expression <right> must "
                    "be a domain that contains the domain that <left> is On. ";
                err_msg += "Collection on the left is On ";
                err_msg += SymbolTable::entitySetName(left_domain);
                err_msg += " and Domain on the right is On ";
                err_msg += SymbolTable::entitySetName(right_domain);
                error(err_msg, node.location());
                return;
            }
        }
    } else {
        // This is for the On operator.
        // Right side must be an entity collection.
        if (!rt.isEntityCollection()) {
            error("in a '<left> On <right>' expression "
                    "the expression <right> must be a Collection Of Cell, Face, Edge or Vertex.", node.location());
            return;
        }
        // Left side must be some collection.
        if (!lt.isCollection()) {
            error("in a '<left> On <right>' expression "
                    "the expression <left> must be a Collection.", node.location());
            return;
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
                error(err_msg, node.location());
                return;
            }
        }
    }
}

void CheckASTVisitor::visit(TrinaryIfNode&)
{
}

void CheckASTVisitor::questionMarkVisit(TrinaryIfNode&)
{
}

void CheckASTVisitor::colonVisit(TrinaryIfNode&)
{
}

void CheckASTVisitor::postVisit(TrinaryIfNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const EquelleType pt = node.predicate()->type();
    const EquelleType tt = node.ifTrue()->type();
    const EquelleType ft = node.ifFalse()->type();
    if (pt.isArray() || tt.isArray() || ft.isArray()) {
        error("in trinary if operator, no operands can be of Array type.", node.location());
        return;
    }
    if (pt.basicType() != Bool) {
        error("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<predicate> must be a Bool type.", node.location());
        return;
    }
    if (tt != ft) {
        error("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<iftrue> and <iffalse> must have the same type.", node.location());
        return;
    }
    if ((pt.isCollection() != tt.isCollection()) ||
        (pt.gridMapping() != tt.gridMapping())) {
        error("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "all three expressions must be 'On' the same set.", node.location());
        return;
    }
    if (node.ifTrue()->dimension() != node.ifFalse()->dimension()) {
        error("in trinary if '<predicate> ? <iftrue> : <iffalse>' "
                "<iftrue> and <iffalse> must have the same dimension.", node.location());
        return;
    }
}

void CheckASTVisitor::visit(VarDeclNode&)
{
}

void CheckASTVisitor::postVisit(VarDeclNode& node)
{
    if (SymbolTable::isVariableDeclared(node.name())
        || SymbolTable::isFunctionDeclared(node.name())) {
        std::string err = "cannot redeclare ";
        err += node.name();
        err += ": already declared.";
        error(err, node.location());
    }
    if (isCheckingSuppressed()) {
        SymbolTable::declareVariable(node.name(), EquelleType());
    } else {
        SymbolTable::declareVariable(node.name(), node.type());
    }
}

void CheckASTVisitor::visit(VarAssignNode&)
{
}

void CheckASTVisitor::postVisit(VarAssignNode& node)
{
    const std::string& name = node.name();
    const ExpressionNode* expr = node.rhs();

    // Check if already declared as function.
    if (SymbolTable::isFunctionDeclared(name)) {
        std::string err_msg = "cannot declare variable ";
        err_msg += name;
        err_msg += ": already declared as function.";
        error(err_msg, node.location());
        return;
    }

    // The remainder of the checks depend on the type,
    // but we do have to declare the variable if not
    // already declared.
    if (isCheckingSuppressed()) {
        // if (!SymbolTable::isVariableDeclared(name)) {
        //     SymbolTable::declareVariable(name, EquelleType());
        //     SymbolTable::setVariableAssigned(name, true);
        // }
        return;
    }

    // If already declared...
    if (SymbolTable::isVariableDeclared(name)) {
        // Check if already assigned.
        if (SymbolTable::isVariableAssigned(name) && !SymbolTable::variableType(name).isMutable()) {
            std::string err_msg = "variable already assigned, cannot re-assign ";
            err_msg += name;
            error(err_msg, node.location());
            return;
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
                error(err_msg, node.location());
                return;
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

    // Set variable to assigned.
    SymbolTable::setVariableAssigned(name, true);
}

void CheckASTVisitor::visit(VarNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    if (SymbolTable::isFunctionDeclared(node.name())) {
        // This is a function reference.
        return;
    }
    if (!SymbolTable::isVariableDeclared(node.name())) {
        std::string err_msg = "using undeclared variable ";
        err_msg += node.name();
        error(err_msg, node.location());
    } else if (!SymbolTable::isVariableAssigned(node.name())) {
        std::string err_msg = "using unassigned variable ";
        err_msg += node.name();
        error(err_msg, node.location());
    }
}

void CheckASTVisitor::visit(FuncRefNode&)
{
}

void CheckASTVisitor::visit(JustAnIdentifierNode&)
{
}

void CheckASTVisitor::visit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsDeclNode&)
{
}

void CheckASTVisitor::visit(FuncDeclNode& node)
{
    SymbolTable::declareFunction(node.name());
    SymbolTable::setCurrentFunction(node.name());
}

void CheckASTVisitor::postVisit(FuncDeclNode& node)
{
    SymbolTable::retypeCurrentFunction(node.ftype()->funcType());
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
}

void CheckASTVisitor::visit(FuncStartNode&)
{
}

void CheckASTVisitor::postVisit(FuncStartNode&)
{
}

void CheckASTVisitor::visit(FuncAssignNode& node)
{
    const bool func_declared = SymbolTable::isFunctionDeclared(node.name());
    if (!func_declared) {
        functemplates_[node.name()] = &node;
        // Get the function argument names (types will be Invalid).
        std::vector<Variable> args;
        auto argnodes = node.args()->arguments();
        args.reserve(argnodes.size());
        for (ExpressionNode* argnode : argnodes) {
            const VarNode& arg = dynamic_cast<const VarNode&>(*argnode);
            args.push_back(arg.name());
        }
        FunctionType ft(args, EquelleType());
        // Declare and set type.
        SymbolTable::declareFunction(node.name(), ft, true);
        // Suppression handling.
        undecl_func_stack.push(node.name());
        suppressChecking();
    }
    SymbolTable::setCurrentFunction(node.name());
}

void CheckASTVisitor::postVisit(FuncAssignNode& node)
{
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
    if (!undecl_func_stack.empty() && undecl_func_stack.top() == node.name()) {
        undecl_func_stack.pop();
        unsuppressChecking();
    }
}

void CheckASTVisitor::visit(FuncArgsNode&)
{
}

void CheckASTVisitor::midVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::postVisit(FuncArgsNode&)
{
}

void CheckASTVisitor::visit(ReturnStatementNode&)
{
}

void CheckASTVisitor::postVisit(ReturnStatementNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    instantiation_return_type_ = node.type();
}

void CheckASTVisitor::visit(FuncCallNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }
    const Function& f = SymbolTable::getFunction(node.name());
    // Check function call arguments.
    const auto& argtypes = node.args()->argumentTypes();
    const auto& fargs = f.functionType().arguments();
    if (argtypes.size() != fargs.size()) {
        std::string err_msg = "wrong number of arguments when calling function ";
        err_msg += node.name();
        error(err_msg, node.location());
    }
    for (int arg = 0; arg < argtypes.size(); ++arg) {
        if (!argtypes[arg].canSubstituteFor(fargs[arg].type())) {
            std::ostringstream err;
            err << "wrong argument type for argument " << arg << " named '"
                << fargs[arg].name() << "' when calling function " << node.name()
                << ", expected " << SymbolTable::equelleString(fargs[arg].type())
                << " but got " << SymbolTable::equelleString(argtypes[arg]);
            error(err.str(), node.location());
        }
    }
    // Special treatment for function templates.
    if (f.isTemplate()) {
        // All the arguments types will be defaulted, so checking in
        // this method is going to succeed automatically (if the
        // correct number of arguments are given). However, we should
        // at this point check that the instantiation makes sense.
        const FunctionType original_ftype = f.functionType();
        const std::set<Variable> original_locvars = f.getLocalVariables();
        const std::string original_scope = SymbolTable::getCurrentFunction().name();
        // Create the function type (except for the return type
        // to be used for the instantiation.
        const auto& args = node.args()->arguments();
        std::vector<Variable> fargs = f.functionType().arguments();
        assert(args.size() == fargs.size());
        const int num_args = fargs.size();
        for (int arg = 0; arg < num_args; ++arg) {
            fargs[arg].setType(args[arg]->type());
            fargs[arg].setAssigned(true);
        }
        FunctionType ft(fargs, EquelleType());
        // Change scope, and verify instantiation by visiting the FuncAssignNode.
        SymbolTable::setCurrentFunction(f.name());
        SymbolTable::retypeCurrentFunction(ft);
        FuncAssignNode* fanode = functemplates_[node.name()];
        instantiation_location_stack_.push_back(node.location());
        fanode->accept(*this);
        instantiation_location_stack_.pop_back();
        // Store instantiated function state. The function has all but
        // the return value in the correct state now.The member
        // instantiation_return_type_ is set in the method
        // postVisit(ReturnStatementNode&).
        Function& fmut = SymbolTable::getMutableFunction(f.name());
        fmut.setReturnType(instantiation_return_type_);
        const int instantiation_index = SymbolTable::addFunctionInstantiation(fmut);
        node.setInstantiationIndex(instantiation_index);
        // Restore function template to its non-instantiated state.
        SymbolTable::setCurrentFunction(f.name());
        SymbolTable::retypeCurrentFunction(original_ftype);
        SymbolTable::getMutableFunction(f.name()).setLocalVariables(original_locvars);
        SymbolTable::setCurrentFunction(original_scope);
        // Set the return type for this node.
        node.setReturnType(instantiation_return_type_);
    }
    // If the function returns a new dynamically created domain,
    // we must declare it (anonymously for now).
    const EquelleType rtype = f.returnType(argtypes);
    if (rtype.isDomain()) {
        const int dynsubret = f.functionType().dynamicSubsetReturn(argtypes);
        if (dynsubret != NotApplicable) {
            // Create a new domain.
            const int gm = SymbolTable::declareNewEntitySet("AnonymousEntitySet", dynsubret);
            node.setDynamicSubsetReturn(gm);
        }
    }
}

void CheckASTVisitor::visit(FuncCallStatementNode&)
{
}

void CheckASTVisitor::postVisit(FuncCallStatementNode&)
{
}

void CheckASTVisitor::visit(LoopNode& node)
{
    // Check that loop_set is a sequence, extract its type.
    const std::string& loop_set = node.loopSet();
    EquelleType loop_set_type;
    if (SymbolTable::isVariableDeclared(loop_set)) {
        if (!isCheckingSuppressed()) {
            loop_set_type = SymbolTable::variableType(loop_set);
            if (!loop_set_type.isSequence()) {
                std::string err_msg = "loop set must be a Sequence: ";
                err_msg += loop_set;
                error(err_msg, node.location());
            }
            if (loop_set_type.isArray()) {
                std::string err_msg = "loop set cannot be an Array: ";
                err_msg += loop_set;
                error(err_msg, node.location());
            }
        }
    } else {
        std::string err_msg = "unknown variable used for loop set: ";
        err_msg += loop_set;
        error(err_msg, node.location());
    }

    // Create a name for the loop scope.
    std::ostringstream os;
    os << "ForLoopWithIndex" << next_loop_index_++;
    // Set name in loop node, declare scope and
    // set to current.
    node.setName(os.str());

    // Declare loop scope and variable.
    SymbolTable::declareFunction(os.str());
    SymbolTable::setCurrentFunction(os.str());
    SymbolTable::declareVariable(node.loopVariable(), loop_set_type.basicType());
    SymbolTable::setVariableAssigned(node.loopVariable(), true);
}

void CheckASTVisitor::postVisit(LoopNode&)
{
    SymbolTable::setCurrentFunction(SymbolTable::getCurrentFunction().parentScope());
}

void CheckASTVisitor::visit(ArrayNode&)
{
}

void CheckASTVisitor::postVisit(ArrayNode& node)
{
    const FuncArgsNode* expr_list = node.expressionList();
    const auto& elems = expr_list->arguments();
    if (elems.empty()) {
        error("cannot create an empty array.", node.location());
        return;
    } else {
        if (!isCheckingSuppressed()) {
            const EquelleType et = elems[0]->type();
            if (et.isArray()) {
                error("an Array cannot contain another Array.", node.location());
                return;
            }
            for (const auto& elem : elems) {
                if (elem->type() != et) {
                    error("elements of an Array must all have the same type", node.location());
                    return;
                }
            }
        }
    }
}

void CheckASTVisitor::visit(RandomAccessNode&)
{
}

void CheckASTVisitor::postVisit(RandomAccessNode& node)
{
    if (isCheckingSuppressed()) {
        return;
    }

    const ExpressionNode* expr = node.expressionToAccess();
    const int index = node.index();
    if (expr->type().isArray()) {
        if (index < 0 || index >= expr->type().arraySize()) {
            error("index out of array bounds in '[<index>]' random access operator.", node.location());
            return;
        }
    } else if (expr->type().basicType() == Vector) {
        if (index < 0 || index > 2) {
            error("cannot use '[<index>]' random access operator on a Vector with index < 0 or > 2", node.location());
            return;
        }
    } else {
        error("cannot use '[<index>]' random access operator with anything other than a Vector or Array", node.location());
        return;
    }
}

void CheckASTVisitor::visit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::midVisit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::postVisit(StencilAssignmentNode&)
{
}

void CheckASTVisitor::visit(StencilNode&)
{
}

void CheckASTVisitor::postVisit(StencilNode&)
{
}

namespace {
    std::ostream& operator<<(std::ostream& os, const FileLocation& loc)
    {
        const int fl = loc.firstLine();
        const int ll = loc.lastLine();
        if (fl == ll) {
            os << "line " << fl;
        } else {
            os << "lines " << fl << '-' << ll;
        }
        return os;
    }
}


void CheckASTVisitor::error(const std::string& err, const FileLocation loc)
{
    std::cerr << "Compile error near " << loc << ": " << err << std::endl;
    if (!instantiation_location_stack_.empty()) {
        auto rbeg = instantiation_location_stack_.rbegin();
        auto rend = instantiation_location_stack_.rend();
        for (auto it = rbeg; it != rend; ++it) {
            std::cerr << "    ---> instantiated from " << *it << std::endl;
        }
    }
}

void CheckASTVisitor::suppressChecking()
{
    ++checking_suppression_level_;
}

void CheckASTVisitor::unsuppressChecking()
{
    --checking_suppression_level_;
}

bool CheckASTVisitor::isCheckingSuppressed() const
{
    return checking_suppression_level_ > 0;
}
