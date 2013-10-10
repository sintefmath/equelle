/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/


#include "Common.hpp"
#include "EquelleType.hpp"
#include "SymbolTable.hpp"

#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <list>
#include <algorithm>
#include <stdexcept>
#include <cassert>



// ============ Methods of EntitySet ============

EntitySet::EntitySet(const int index, const int subset_index)
    : index_(index), subset_index_(subset_index)
{
}

int EntitySet::index() const
{
    return index_;
}

int EntitySet::subsetIndex() const
{
    return subset_index_;
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

EquelleType FunctionType::returnType(const std::vector<EquelleType>& argtypes) const
{
    if (dynamic_.active) {
        const BasicType bt = dynamic_.arg_index_for_basic_type == InvalidIndex ?
            return_type_.basicType() : argtypes[dynamic_.arg_index_for_basic_type].basicType();
        const bool coll = return_type_.isCollection();
        const int gridmapping = dynamic_.arg_index_for_gridmapping == InvalidIndex ?
            return_type_.gridMapping() : argtypes[dynamic_.arg_index_for_gridmapping].gridMapping();
        return EquelleType(bt, coll, gridmapping);
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
                NotApplicable : argtypes[dynamic_.arg_index_for_subset].gridMapping();
            return subset;
        }
    }
    return NotApplicable;
}

const std::vector<Variable>& FunctionType::arguments() const
{
    return arguments_;
}



// ============ Methods of Function ============


Function::Function(const std::string& name, const FunctionType& type)
    : name_(name),
      type_(type)
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

int Function::declareEntitySet(const int new_entity_index, const int subset_entity_index)
{
    local_entitysets_.emplace_back(new_entity_index, subset_entity_index);
    return new_entity_index;
}

EquelleType Function::variableType(const std::string name) const
{
    auto foundvar = declared(name);
    if (!foundvar.first) {
        std::string err_msg = "could not find variable ";
        err_msg += name;
        yyerror(err_msg.c_str());
        return EquelleType();
    } else {
        return foundvar.second;
    }
}

bool Function::isVariableDeclared(const std::string& name) const
{
    return declared(name).first;
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
            yyerror("internal compiler error in Function::isVariableAssigned()");
            return false;
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

bool Function::isSubset(const int set1, const int set2) const
{
    if (set1 == set2) {
        return true;
    }
    auto it = findSet(set1);
    if (it == local_entitysets_.end()) {
        yyerror("internal compiler error in Function::isSubset()");
        return false;
    }
    if (it->subsetIndex() == set2) {
        return true;
    }
    if (it->subsetIndex() == set1) {
        return false;
    }
    return isSubset(it->subsetIndex(), set2);
}

std::string Function::name() const
{
    return name_;
}

const FunctionType& Function::functionType() const
{
    return type_;
}

EquelleType Function::returnType(const std::vector<EquelleType>& argtypes) const
{
    return type_.returnType(argtypes);
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

std::vector<EntitySet>::const_iterator Function::findSet(const int index) const
{
    return std::find_if(local_entitysets_.begin(), local_entitysets_.end(),
                        [&](const EntitySet& es) { return es.index() == index; });
}



// ============ Methods of SymbolTable ============


void SymbolTable::declareVariable(const std::string& name, const EquelleType& type)
{
    instance().current_function_->declareVariable(name, type);
}

void SymbolTable::declareFunction(const std::string& name, const FunctionType& ftype)
{
    instance().declareFunctionImpl(name, ftype);
}

int SymbolTable::declareNewEntitySet(const int subset_entity_index)
{
    return instance().current_function_->declareEntitySet(instance().next_subset_index_++, subset_entity_index);
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
    return instance().current_function_->setVariableType(name, type);
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

/// Returns true if set1 is a (non-strict) subset of set2.
bool SymbolTable::isSubset(const int set1, const int set2)
{
    return instance().current_function_->isSubset(set1, set2);
}

SymbolTable::SymbolTable()
    : next_subset_index_(FirstRuntimeEntitySet)
{
    // ----- Add built-in functions to function table. -----
    // 1. Grid functions.
    functions_.emplace_back("Main", FunctionType(EquelleType()));
    functions_.emplace_back("InteriorCells", FunctionType(EquelleType(Cell, true, InteriorCells)));
    functions_.emplace_back("BoundaryCells", FunctionType(EquelleType(Cell, true, BoundaryCells)));
    functions_.emplace_back("AllCells", FunctionType(EquelleType(Cell, true, AllCells)));
    functions_.emplace_back("InteriorFaces", FunctionType(EquelleType(Cell, true, InteriorFaces)));
    functions_.emplace_back("BoundaryFaces", FunctionType(EquelleType(Cell, true, BoundaryFaces)));
    functions_.emplace_back("AllFaces", FunctionType(EquelleType(Cell, true, AllFaces)));
    functions_.emplace_back("InteriorEdges", FunctionType(EquelleType(Cell, true, InteriorEdges)));
    functions_.emplace_back("BoundaryEdges", FunctionType(EquelleType(Cell, true, BoundaryEdges)));
    functions_.emplace_back("AllEdges", FunctionType(EquelleType(Cell, true, AllEdges)));
    functions_.emplace_back("InteriorVertices", FunctionType(EquelleType(Cell, true, InteriorVertices)));
    functions_.emplace_back("BoundaryVertices", FunctionType(EquelleType(Cell, true, BoundaryVertices)));
    functions_.emplace_back("AllVertices", FunctionType(EquelleType(Cell, true, AllVertices)));
    functions_.emplace_back("FirstCell",
                            FunctionType({ Variable("faces", EquelleType()) },
                                         EquelleType(Cell, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("SecondCell",
                            FunctionType({ Variable("faces", EquelleType()) },
                                         EquelleType(Cell, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("IsEmpty",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Bool, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("Centroid",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Vector, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("Normal",
                            FunctionType({ Variable("faces", EquelleType()) },
                                         EquelleType(Vector, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    // 2. User input functions.
    functions_.emplace_back("UserSpecifiedScalarWithDefault",
                            FunctionType({ Variable("default", EquelleType(Scalar)) },
                                         EquelleType(Scalar)));
    functions_.emplace_back("UserSpecifiedCollectionOfScalar",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Scalar, true),
                                         { InvalidIndex, 0, InvalidIndex}));
    functions_.emplace_back("UserSpecifiedCollectionOfFaceSubsetOf",
                            FunctionType({ Variable("entities", EquelleType()) },
                                         EquelleType(Face, true),
                                         { InvalidIndex, InvalidIndex, 0}));
    // 3. Discrete operators.
    functions_.emplace_back("Gradient",
                            FunctionType({ Variable("values", EquelleType(Scalar, true, AllCells)) },
                                         EquelleType(Scalar, true, InteriorFaces)));
    functions_.emplace_back("Divergence",
                            FunctionType({ Variable("values", EquelleType()) },
                                         EquelleType(Scalar, true, AllCells)));
    // 4. Other functions
    functions_.emplace_back("NewtonSolve",
                            FunctionType({ Variable("u", EquelleType()) },
                                         EquelleType(Scalar, true, AllCells)));
    functions_.emplace_back("Output",
                            FunctionType({ Variable("data", EquelleType()) },
                                         EquelleType()));


    // ----- Set main function ref and current (initially equal to main). -----
    main_function_ = functions_.begin();
    current_function_ = main_function_;

    // ----- Add built-in entity sets to entity set table. -----
    main_function_->declareEntitySet(InteriorCells, AllCells);
    main_function_->declareEntitySet(BoundaryCells, AllCells);
    main_function_->declareEntitySet(AllCells, AllCells);
    main_function_->declareEntitySet(InteriorFaces, AllFaces);
    main_function_->declareEntitySet(BoundaryFaces, AllFaces);
    main_function_->declareEntitySet(AllFaces, AllFaces);
    main_function_->declareEntitySet(InteriorEdges, AllEdges);
    main_function_->declareEntitySet(BoundaryEdges, AllEdges);
    main_function_->declareEntitySet(AllEdges, AllEdges);
    main_function_->declareEntitySet(InteriorVertices, AllVertices);
    main_function_->declareEntitySet(BoundaryVertices, AllVertices);
    main_function_->declareEntitySet(AllVertices, AllVertices);
}

SymbolTable& SymbolTable::instance()
{
    static SymbolTable s;
    return s;
}

/// Used only for setting up initial built-in entity sets.
int SymbolTable::declareEntitySet(const int entity_index, const int subset_entity_index)
{
    return instance().current_function_->declareEntitySet(entity_index, subset_entity_index);
}

void SymbolTable::declareFunctionImpl(const std::string& name, const FunctionType& ftype)
{
    if (current_function_->name() != "Main") {
        std::string err_msg = "cannot define new function ";
        err_msg += name;
        err_msg += " inside function ";
        err_msg += current_function_->name();
        yyerror(err_msg.c_str());
    } else {
        auto it = findFunction(name);
        if (it == functions_.end()) {
            functions_.emplace_back(name, ftype);
        }
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

std::list<Function>::const_iterator SymbolTable::findFunction(const std::string& name) const
{
    auto it = std::find_if(functions_.begin(), functions_.end(),
                           [&](const Function& f) { return f.name() == name; });
    return it;
}
