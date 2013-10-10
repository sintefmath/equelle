/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef SYMBOLTABLE_HEADER_INCLUDED
#define SYMBOLTABLE_HEADER_INCLUDED


#include "Common.hpp"
#include "EquelleType.hpp"

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

// ------ SymbolTable singleton and classes used by it. ------ 



class EntitySet
{
public:
    EntitySet(const int index, const int subset_index);
    int index() const;
    int subsetIndex() const;

private:
    int index_;
    int subset_index_;
};




class Variable
{
public:
    Variable(const std::string& name,
             const EquelleType type = EquelleType(),
             bool assigned = false);
    const std::string& name() const;
    const EquelleType& type() const;
    void setType(const EquelleType& type);
    bool assigned() const;
    void setAssigned(const bool assigned);
    bool operator<(const Variable& v) const;
private:
    std::string name_;
    EquelleType type_;
    bool assigned_;
};




enum { InvalidIndex = -1 };


struct DynamicReturnSpecification
{
    DynamicReturnSpecification()
        : active(false),
          arg_index_for_basic_type(InvalidIndex),
          arg_index_for_gridmapping(InvalidIndex),
          arg_index_for_subset(InvalidIndex)
    {
    }
    DynamicReturnSpecification(const int bt_ix, const int gm_ix, const int ss_ix)
        : active(true),
          arg_index_for_basic_type(bt_ix),
          arg_index_for_gridmapping(gm_ix),
          arg_index_for_subset(ss_ix)
    {
    }
    bool active;
    int arg_index_for_basic_type;
    int arg_index_for_gridmapping;
    int arg_index_for_subset;
};




class FunctionType
{
public:
    /// Construct FunctionType taking no arguments.
    /// Equelle type: Function() -> returntype
    explicit FunctionType(const EquelleType& return_type);

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type);

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type,
                 const DynamicReturnSpecification& dynamic);

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const;

    int dynamicSubsetReturn(const std::vector<EquelleType>& argtypes) const;

    const std::vector<Variable>& arguments() const;

private:
    std::vector<Variable> arguments_;
    EquelleType return_type_;
    DynamicReturnSpecification dynamic_;
};




class Function
{
public:
    Function(const std::string& name, const FunctionType& type);

    void declareVariable(const std::string& name, const EquelleType& type);

    int declareEntitySet(const int new_entity_index, const int subset_entity_index);

    EquelleType variableType(const std::string name) const;

    bool isVariableDeclared(const std::string& name) const;

    bool isVariableAssigned(const std::string& name) const;

    void setVariableAssigned(const std::string& name, const bool assigned);

    void setVariableType(const std::string& name, const EquelleType& type);

    bool isSubset(const int set1, const int set2) const;

    std::string name() const;

    const FunctionType& functionType() const;

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const;

private:
    std::pair<bool, EquelleType> declared(const std::string& name) const;

    std::vector<EntitySet>::const_iterator findSet(const int index) const;

    std::string name_;
    std::set<Variable> local_variables_;
    std::vector<EntitySet> local_entitysets_;
    FunctionType type_;
};




class SymbolTable
{
public:
    static void declareVariable(const std::string& name, const EquelleType& type);

    static void declareFunction(const std::string& name, const FunctionType& ftype);

    static int declareNewEntitySet(const int subset_entity_index);

    static bool isVariableDeclared(const std::string& name);

    static bool isVariableAssigned(const std::string& name);

    static void setVariableAssigned(const std::string& name, const bool assigned);

    static EquelleType variableType(const std::string& name);

    static void setVariableType(const std::string& name, const EquelleType& type);

    static bool isFunctionDeclared(const std::string& name);

    static const Function& getFunction(const std::string& name);

    static const Function& getCurrentFunction();

    /// Returns true if set1 is a (non-strict) subset of set2.
    static bool isSubset(const int set1, const int set2);

private:
    SymbolTable();

    static SymbolTable& instance();

    /// Used only for setting up initial built-in entity sets.
    static int declareEntitySet(const int entity_index, const int subset_entity_index);

    void declareFunctionImpl(const std::string& name, const FunctionType& ftype);

    bool isFunctionDeclaredImpl(const std::string& name) const;

    const Function& getFunctionImpl(const std::string& name) const;

    std::list<Function>::const_iterator findFunction(const std::string& name) const;

    int next_subset_index_;
    std::list<Function> functions_;
    std::list<Function>::iterator main_function_;
    std::list<Function>::iterator current_function_;
};





#endif // SYMBOLTABLE_HEADER_INCLUDED
