/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "SymbolTable.hpp"
#include "Common.hpp"
#include "EquelleType.hpp"
#include "NodeInterface.hpp"

#include <algorithm>
#include <stdexcept>
#include <iostream>



// ============ Methods of EntitySet ============

EntitySet::EntitySet(const std::string& name, const int index, const int subset_index)
    : name_(name), index_(index), subset_index_(subset_index)
{
}

const std::string& EntitySet::name() const
{
    return name_;
}

int EntitySet::index() const
{
    return index_;
}

int EntitySet::subsetIndex() const
{
    return subset_index_;
}

void EntitySet::setName(const std::string& name)
{
    name_ = name;
}


// ============ Methods of Variable ============


Variable::Variable(const std::string& name,
                   const EquelleType type,
                   bool assigned)
    : name_(name), type_(type), assigned_(assigned)
{
}

const std::string& Variable::name() const
{
    return name_;
}

const EquelleType& Variable::type() const
{
    return type_;
}

void Variable::setType(const EquelleType& type)
{
    type_ = type;
}

bool Variable::assigned() const
{
    return assigned_;
}

void Variable::setAssigned(const bool assigned)
{
    assigned_ = assigned;
}

bool Variable::operator<(const Variable& v) const
{
    return name_ < v.name_;
}



// ============ Methods of FunctionType ============


FunctionType::FunctionType()
{
}

/// Construct FunctionType taking no arguments.
/// Equelle type: Function() -> returntype
FunctionType::FunctionType(const EquelleType& return_type)
    : return_type_(return_type)
{
}

FunctionType::FunctionType(const std::vector<Variable>& args,
                           const EquelleType& return_type)
    : arguments_(args),
      return_type_(return_type)
{
}

FunctionType::FunctionType(const std::vector<Variable>& args,
                           const EquelleType& return_type,
                           const DynamicReturnSpecification& dynamic)
    : arguments_(args),
      return_type_(return_type),
      dynamic_(dynamic)
{
}

EquelleType FunctionType::returnType() const
{
    if (dynamic_.active) {
        throw std::logic_error("Should not call FunctionType::returnType() with no arguments "
                               "when the function has dynamic return type.");
    } else {
        return return_type_;
    }
}

EquelleType FunctionType::returnType(const std::vector<EquelleType>& argtypes) const
{
    if (dynamic_.active) {
        const BasicType bt = dynamic_.arg_index_for_basic_type == InvalidIndex ?
            return_type_.basicType() : argtypes[dynamic_.arg_index_for_basic_type].basicType();
        const int gridmapping = dynamic_.arg_index_for_gridmapping == InvalidIndex ?
            return_type_.gridMapping() : argtypes[dynamic_.arg_index_for_gridmapping].gridMapping();
        const int subset = dynamicSubsetReturn(argtypes);
        const int array_size = dynamic_.arg_index_for_array_size == InvalidIndex ?
            return_type_.arraySize() : argtypes[dynamic_.arg_index_for_array_size].arraySize();
        return EquelleType(bt, return_type_.compositeType(), gridmapping, subset, false, return_type_.isDomain(), array_size);
    } else {
        return return_type_;
    }
}

int FunctionType::dynamicSubsetReturn(const std::vector<EquelleType>& argtypes) const
{
    if (dynamic_.active) {
        const BasicType bt = dynamic_.arg_index_for_basic_type == InvalidIndex ?
            return_type_.basicType() : argtypes[dynamic_.arg_index_for_basic_type].basicType();
        const bool coll = return_type_.isCollection();
        if (isEntityType(bt) && coll) {
            const int subset = dynamic_.arg_index_for_subset == InvalidIndex ?
                return_type_.subsetOf() : argtypes[dynamic_.arg_index_for_subset].gridMapping();
            return subset;
        }
    }
    return NotApplicable;
}

const std::vector<Variable>& FunctionType::arguments() const
{
    return arguments_;
}


std::string FunctionType::equelleString() const
{
    std::string retval = "Function(";
    for (auto var : arguments_) {
        retval += var.name();
        retval += " : ";
        if (var.type().basicType() == Invalid) {
            retval += " <multiple types possible>";
        } else {
            retval += SymbolTable::equelleString(var.type());
        }
        retval += ", ";
    }
    if (!arguments_.empty()) {
        // Chop the extra comma and space.
        retval.erase(retval.end() - 2, retval.end());
    }
    retval += ") -> ";
    retval += SymbolTable::equelleString(return_type_);
    return retval;
}


// ============ Methods of Function ============


Function::Function(const std::string& name)
    : name_(name),
      parent_scope_(0)
{
}

Function::Function(const std::string& name, const FunctionType& type)
    : name_(name),
      type_(type),
      parent_scope_(0)
{
}

void Function::declareVariable(const std::string& name, const EquelleType& type)
{
    if (!declared(name).first) {
        local_variables_.insert(Variable(name, type, false));
    } else {
        std::string errmsg = "redeclared variable: ";
        errmsg += name;
        yyerror(errmsg.c_str());
    }
}

EquelleType Function::variableType(const std::string name) const
{
    auto foundvar = declared(name);
    if (!foundvar.first) {
        if (parent_scope_) {
            return parent_scope_->variableType(name);
        } else {
            std::string err_msg = "could not find variable ";
            err_msg += name;
            yyerror(err_msg.c_str());
            return EquelleType();
        }
    } else {
        return foundvar.second;
    }
}

bool Function::isVariableDeclared(const std::string& name) const
{
    auto it = declared(name);
    if (it.first) {
        return true;
    } else if (parent_scope_) {
        return parent_scope_->isVariableDeclared(name);
    } else {
        return false;
    }
}

bool Function::isVariableAssigned(const std::string& name) const
{
    auto lit = local_variables_.find(Variable(name));
    if (lit != local_variables_.end()) {
        return lit->assigned();
    } else {
        auto ait = std::find_if(type_.arguments().begin(), type_.arguments().end(),
                                [&](const Variable& a) { return a.name() == name; });
        if (ait != type_.arguments().end()) {
            return ait->assigned();
        } else {
            if (parent_scope_) {
                return parent_scope_->isVariableAssigned(name);
            } else {
                yyerror("internal compiler error in Function::isVariableAssigned()");
                return false;
            }
        }
    }
}

void Function::setVariableAssigned(const std::string& name, const bool assigned)
{
    auto lit = local_variables_.find(Variable(name));
    if (lit != local_variables_.end()) {
        // Set members are immutable, must
        // copy, erase and reinsert.
        Variable copy = *lit;
        copy.setAssigned(assigned);
        local_variables_.erase(lit);
        local_variables_.insert(copy);
    } else {
        yyerror("internal compiler error in Function::setVariableAssigned()");
    }
}

void Function::setVariableType(const std::string& name, const EquelleType& type)
{
    auto lit = local_variables_.find(Variable(name));
    if (lit != local_variables_.end()) {
        // Set members are immutable, must
        // copy, erase and reinsert.
        Variable copy = *lit;
        copy.setType(type);
        local_variables_.erase(lit);
        local_variables_.insert(copy);
    } else {
        yyerror("internal compiler error in Function::setVariableType()");
    }
}

const std::string& Function::name() const
{
    return name_;
}

void Function::setName(const std::string& name)
{
    name_ = name;
}

const FunctionType& Function::functionType() const
{
    return type_;
}

void Function::setFunctionType(const FunctionType& ftype)
{
    type_ = ftype;
}

EquelleType Function::returnType(const std::vector<EquelleType>& argtypes) const
{
    return type_.returnType(argtypes);
}

void Function::setParentScope(const Function* parent_scope)
{
    parent_scope_ = parent_scope;
}

const std::string& Function::parentScope() const
{
    if (parent_scope_) {
        return parent_scope_->name();
    } else {
        throw std::runtime_error("Internal compiler error in Function::parentScope().");
    }
}

void Function::dump() const
{
    std::cout << "------------------ Dump of function: " << name() << " ------------------\n";
    std::cout << type_.equelleString() << '\n';
    std::cout << "Local variables:\n";
    for (const Variable& v : local_variables_) {
        std::cout << v.name() << " : " << SymbolTable::equelleString(v.type()) << "    assigned: " << v.assigned() << '\n';
    }
    if (parent_scope_) {
        std::cout << "Parent scope is: " << parent_scope_->name() << '\n';
    }
}

std::pair<bool, EquelleType> Function::declared(const std::string& name) const
{
    auto lit = local_variables_.find(name);
    if (lit != local_variables_.end()) {
        return std::make_pair(true, lit->type());
    }
    auto ait = std::find_if(type_.arguments().begin(), type_.arguments().end(),
                            [&](const Variable& a) { return a.name() == name; });
    if (ait != type_.arguments().end()) {
        return std::make_pair(true, ait->type());
    }
    return std::make_pair(false, EquelleType());
}



// ============ Methods of SymbolTable ============


void SymbolTable::declareVariable(const std::string& name, const EquelleType& type)
{
    instance().current_function_->declareVariable(name, type);
}

void SymbolTable::declareFunction(const std::string& name)
{
    instance().declareFunctionImpl(name, FunctionType());
}

void SymbolTable::declareFunction(const std::string& name, const FunctionType& ftype)
{
    instance().declareFunctionImpl(name, ftype);
}

int SymbolTable::declareNewEntitySet(const std::string& name, const int subset_entity_index)
{
    const int new_entityset_index = instance().next_entityset_index_++;
    instance().declareEntitySet(name, new_entityset_index, subset_entity_index);
    return new_entityset_index;
}

bool SymbolTable::isVariableDeclared(const std::string& name)
{
    return instance().current_function_->isVariableDeclared(name);
}

bool SymbolTable::isVariableAssigned(const std::string& name)
{
    return instance().current_function_->isVariableAssigned(name);
}

void SymbolTable::setVariableAssigned(const std::string& name, const bool assigned)
{
    return instance().current_function_->setVariableAssigned(name, assigned);
}

EquelleType SymbolTable::variableType(const std::string& name)
{
    return instance().current_function_->variableType(name);
}

void SymbolTable::setVariableType(const std::string& name, const EquelleType& type)
{
    instance().current_function_->setVariableType(name, type);
}

bool SymbolTable::isFunctionDeclared(const std::string& name)
{
    return instance().isFunctionDeclaredImpl(name);
}

const Function& SymbolTable::getFunction(const std::string& name)
{
    return instance().getFunctionImpl(name);
}

const Function& SymbolTable::getCurrentFunction()
{
    return *instance().current_function_;
}

void SymbolTable::setCurrentFunction(const std::string& name)
{
    instance().setCurrentFunctionImpl(name);
}

void SymbolTable::renameCurrentFunction(const std::string& name)
{
    instance().current_function_->setName(name);
}

void SymbolTable::retypeCurrentFunction(const FunctionType& ftype)
{
    instance().current_function_->setFunctionType(ftype);
}

/// Returns true if set1 is a (non-strict) subset of set2.
bool SymbolTable::isSubset(const int set1, const int set2)
{
    return instance().isSubsetImpl(set1, set2);
}

Node* SymbolTable::program()
{
    return instance().ast_root_;
}

void SymbolTable::setProgram(Node* ast_root)
{
    instance().ast_root_ = ast_root;
}

std::string SymbolTable::equelleString(const EquelleType& type)
{
    std::string retval;
    if (type.isMutable()) {
        retval += "Mutable ";
    }
    if (type.isArray()) {
        retval += "Array Of " + std::to_string(type.arraySize()) + " ";
    }
    if (type.isCollection()) {
        retval += "Collection Of ";
    } else if (type.isSequence()) {
        retval += "Sequence Of ";
    }
    retval += basicTypeString(type.basicType());
    if (type.gridMapping() != NotApplicable
        && type.gridMapping() != PostponedDefinition) {
        retval += " On ";
        retval += instance().findSet(type.gridMapping())->name();
    }
    if (type.subsetOf() != NotApplicable) {
        retval += " Subset Of ";
        retval += instance().findSet(type.subsetOf())->name();
    }
    return retval;
}

const std::string& SymbolTable::entitySetName(const int entity_set_index)
{
    return instance().findSet(entity_set_index)->name();
}

BasicType SymbolTable::entitySetType(const int entity_set_index)
{
    BasicType canonical = canonicalGridMappingEntity(entity_set_index);
    int es = entity_set_index;
    while (canonical == Invalid) {
        es = instance().findSet(es)->subsetIndex();
        canonical = canonicalGridMappingEntity(es);
    }
    return canonical;
}

void SymbolTable::setEntitySetName(const int entity_set_index, const std::string& name)
{
    instance().findSet(entity_set_index)->setName(name);
}

void SymbolTable::dump()
{
    instance().dumpImpl();
}

SymbolTable::SymbolTable()
    : next_entityset_index_(FirstRuntimeEntitySet),
      ast_root_(nullptr)
{
    // ----- Add built-in functions to function table. -----
    // 1. Grid functions.
    functions_.emplace_back("Main", FunctionType(EquelleType(Void)));
    functions_.emplace_back("InteriorCells", FunctionType(EquelleType(Cell, Collection, InteriorCells, AllCells, false, true)));
    functions_.emplace_back("BoundaryCells", FunctionType(EquelleType(Cell, Collection, BoundaryCells, AllCells, false, true)));
    functions_.emplace_back("AllCells", FunctionType(EquelleType(Cell, Collection, AllCells, AllCells, false, true)));
    functions_.emplace_back("InteriorFaces", FunctionType(EquelleType(Face, Collection, InteriorFaces, AllFaces, false, true)));
    functions_.emplace_back("BoundaryFaces", FunctionType(EquelleType(Face, Collection, BoundaryFaces, AllFaces, false, true)));
    functions_.emplace_back("AllFaces", FunctionType(EquelleType(Face, Collection, AllFaces, AllFaces, false, true)));
    functions_.emplace_back("InteriorEdges", FunctionType(EquelleType(Edge, Collection, InteriorEdges, AllEdges, false, true)));
    functions_.emplace_back("BoundaryEdges", FunctionType(EquelleType(Edge, Collection, BoundaryEdges, AllEdges, false, true)));
    functions_.emplace_back("AllEdges", FunctionType(EquelleType(Edge, Collection, AllEdges, AllEdges, false, true)));
    functions_.emplace_back("InteriorVertices", FunctionType(EquelleType(Vertex, Collection, InteriorVertices, AllVertices, false, true)));
    functions_.emplace_back("BoundaryVertices", FunctionType(EquelleType(Vertex, Collection, BoundaryVertices, AllVertices, false, true)));
    functions_.emplace_back("AllVertices", FunctionType(EquelleType(Vertex, Collection, AllVertices, AllVertices, false, true)));
    functions_.emplace_back("FirstCell",
                            FunctionType({ Variable("faces", EquelleType(Face, Collection)) },
                                         EquelleType(Cell, Collection, NotApplicable, AllCells),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("SecondCell",
                            FunctionType({ Variable("faces", EquelleType(Face, Collection)) },
                                         EquelleType(Cell, Collection, NotApplicable, AllCells),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("IsEmpty",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Bool, Collection),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("Centroid",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Vector, Collection),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("Normal",
                            FunctionType({ Variable("faces", EquelleType(Face, Collection)) },
                                         EquelleType(Vector, Collection),
                                         { InvalidIndex, 0, InvalidIndex}));
    // 2. User input functions.
    functions_.emplace_back("InputScalarWithDefault",
                            FunctionType({ Variable("name", EquelleType(String)),
                                           Variable("default", EquelleType(Scalar)) },
                                         EquelleType(Scalar)));
    functions_.emplace_back("InputCollectionOfScalar",
                            FunctionType({ Variable("name", EquelleType(String)),
                                           Variable("entities", EquelleType()) },
                                         EquelleType(Scalar, Collection),
                                         { InvalidIndex, 1, InvalidIndex}));
    functions_.emplace_back("InputDomainSubsetOf",
                            FunctionType({ Variable("name", EquelleType(String)),
                                           Variable("entities", EquelleType()) },
                                          EquelleType(Invalid, Collection, NotApplicable, NotApplicable, false, true),
                                         { 1, InvalidIndex, 1}));
    functions_.emplace_back("InputSequenceOfScalar",
                            FunctionType({ Variable("name", EquelleType(String)) },
                                         EquelleType(Scalar, Sequence)));


    // 3. Discrete operators.
    functions_.emplace_back("Gradient",
                            FunctionType({ Variable("values", EquelleType(Scalar, Collection, AllCells)) },
                                         EquelleType(Scalar, Collection, InteriorFaces)));
    functions_.emplace_back("Divergence",
                            FunctionType({ Variable("values", EquelleType(Scalar, Collection)) },
                                         EquelleType(Scalar, Collection, AllCells)));
    // 4. Other functions
    functions_.emplace_back("Dot",
                            FunctionType({ Variable("v1", EquelleType(Vector, Collection)),
                                           Variable("v2", EquelleType(Vector, Collection)) },
                                EquelleType(Scalar, Collection),
                                {InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("NewtonSolve",
                            FunctionType({ Variable("residual_function", EquelleType()),
                                           Variable("u_guess", EquelleType(Scalar, Collection)) },
                                EquelleType(Scalar, Collection),
                                {InvalidIndex, 1, InvalidIndex}));
    functions_.emplace_back("NewtonSolveSystem",
                            FunctionType({ Variable("residual_function_array", EquelleType()),
                                           Variable("u_guess_array", EquelleType(Scalar, Collection)) },
                                EquelleType(Scalar, Collection),
                                {InvalidIndex, 1, InvalidIndex, 1}));
    functions_.emplace_back("Output",
                            FunctionType({ Variable("tag", EquelleType(String)),
                                           Variable("data", EquelleType()) },
                                         EquelleType(Void)));
    functions_.emplace_back("Sqrt",
                            FunctionType({ Variable("s", EquelleType(Scalar, Collection)) },
                                EquelleType(Scalar, Collection),
                                {InvalidIndex, 0, InvalidIndex}));

    functions_.emplace_back("MaxReduce",
                            FunctionType({ Variable("x", EquelleType(Scalar, Collection)) },
                                         EquelleType(Scalar)));

    functions_.emplace_back("MinReduce",
                            FunctionType({ Variable("x", EquelleType(Scalar, Collection)) },
                                         EquelleType(Scalar)));

    functions_.emplace_back("SumReduce",
                            FunctionType({ Variable("x", EquelleType(Scalar, Collection)) },
                                         EquelleType(Scalar)));

    functions_.emplace_back("ProdReduce",
                            FunctionType({ Variable("x", EquelleType(Scalar, Collection)) },
                                         EquelleType(Scalar)));

    functions_.emplace_back("StencilI",
                            FunctionType( EquelleType( StencilI ) ) );
    functions_.emplace_back("StencilJ",
                                FunctionType( EquelleType( StencilJ ) ) );


    // ----- Set main function ref and current (initially equal to main). -----
    main_function_ = functions_.begin();
    current_function_ = main_function_;

    // ----- Add built-in entity sets to entity set table. -----
    declareEntitySet("InteriorCells()", InteriorCells, AllCells);
    declareEntitySet("BoundaryCells()",BoundaryCells, AllCells);
    declareEntitySet("AllCells()", AllCells, AllCells);
    declareEntitySet("InteriorFaces()", InteriorFaces, AllFaces);
    declareEntitySet("BoundaryFaces()", BoundaryFaces, AllFaces);
    declareEntitySet("AllFaces()", AllFaces, AllFaces);
    declareEntitySet("InteriorEdges()", InteriorEdges, AllEdges);
    declareEntitySet("BoundaryEdges()", BoundaryEdges, AllEdges);
    declareEntitySet("AllEdges()", AllEdges, AllEdges);
    declareEntitySet("InteriorVertices()", InteriorVertices, AllVertices);
    declareEntitySet("BoundaryVertices()", BoundaryVertices, AllVertices);
    declareEntitySet("AllVertices()", AllVertices, AllVertices);
}

SymbolTable::~SymbolTable()
{
    delete ast_root_; // OK even if null.
}

/// Using the Meyers singleton pattern.
SymbolTable& SymbolTable::instance()
{
    static SymbolTable s;
    return s;
}

/// Used only for setting up initial built-in entity sets.
void SymbolTable::declareEntitySet(const std::string& name, const int entity_index, const int subset_entity_index)
{
    entitysets_.emplace_back(name, entity_index, subset_entity_index);
}

void SymbolTable::declareFunctionImpl(const std::string& name, const FunctionType& ftype)
{
    auto it = findFunction(name);
    if (it == functions_.end()) {
        functions_.emplace_back(name, ftype);
        functions_.back().setParentScope(&*current_function_);
    } else {
        std::string errmsg = "function already declared: ";
        errmsg += name;
        yyerror(errmsg.c_str());
        throw std::logic_error("Function already declared.");
    }
}

bool SymbolTable::isFunctionDeclaredImpl(const std::string& name) const
{
    return findFunction(name) != functions_.end();
}

const Function& SymbolTable::getFunctionImpl(const std::string& name) const
{
    auto it = findFunction(name);
    if (it == functions_.end()) {
        std::string errmsg = "could not find function ";
        errmsg += name;
        yyerror(errmsg.c_str());
        throw std::logic_error("Function not found.");
    } else {
        return *it;
    }
}

void SymbolTable::setCurrentFunctionImpl(const std::string& name)
{
    auto func = findFunction(name);
    if (func == functions_.end()) {
        std::string err_msg("internal compiler error: could not find function ");
        err_msg += name;
        yyerror(err_msg.c_str());
    } else {
        current_function_ = func;
    }
}

bool SymbolTable::isSubsetImpl(const int set1, const int set2) const
{
    if (set1 == set2) {
        return true;
    }
    auto it = findSet(set1);
    if (it == entitysets_.end()) {
        yyerror("internal compiler error in Function::isSubset()");
        return false;
    }
    if (it->subsetIndex() == set2) {
        return true;
    }
    if (it->subsetIndex() == set1) {
        return false;
    }
    return isSubsetImpl(it->subsetIndex(), set2);
}

void SymbolTable::dumpImpl() const
{
    std::cout << "================== Dump of symbol table ==================\n";
    for (const Function& f : functions_) {
        f.dump();
    }
    std::cout << "================== End of symbol table dump ==================\n";
}


std::list<Function>::iterator SymbolTable::findFunction(const std::string& name)
{
    auto it = std::find_if(functions_.begin(), functions_.end(),
                           [&](const Function& f) { return f.name() == name; });
    return it;
}

std::list<Function>::const_iterator SymbolTable::findFunction(const std::string& name) const
{
    auto it = std::find_if(functions_.begin(), functions_.end(),
                           [&](const Function& f) { return f.name() == name; });
    return it;
}


std::vector<EntitySet>::iterator SymbolTable::findSet(const int index)
{
    return std::find_if(entitysets_.begin(), entitysets_.end(),
                        [&](const EntitySet& es) { return es.index() == index; });
}

std::vector<EntitySet>::const_iterator SymbolTable::findSet(const int index) const
{
    return std::find_if(entitysets_.begin(), entitysets_.end(),
                        [&](const EntitySet& es) { return es.index() == index; });
}
