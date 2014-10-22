/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef SYMBOLTABLE_HEADER_INCLUDED
#define SYMBOLTABLE_HEADER_INCLUDED


#include "Common.hpp"
#include "EquelleType.hpp"
#include "Dimension.hpp"

#include <set>
#include <vector>
#include <list>


// Must forward-declare the AST node type for SymbolType::program() and SymbolTable::setProgram().
class Node;


// ------ SymbolTable singleton and classes used by it. ------ 



class EntitySet
{
public:
    EntitySet(const std::string& name, const int index, const int subset_index);
    const std::string& name() const;
    int index() const;
    int subsetIndex() const;
    void setName(const std::string& name);
private:
    std::string name_;
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
    const Dimension& dimension() const;
    void setDimension(const Dimension& dimension);
    bool operator<(const Variable& v) const;
private:
    std::string name_;
    EquelleType type_;
    Dimension dimension_;
    bool assigned_;
};




enum { InvalidIndex = -1 };


struct DynamicReturnSpecification
{
    DynamicReturnSpecification()
        : active(false),
          arg_index_for_basic_type(InvalidIndex),
          arg_index_for_gridmapping(InvalidIndex),
          arg_index_for_subset(InvalidIndex),
          arg_index_for_array_size(InvalidIndex),
          arg_index_for_dimension(InvalidIndex)
    {
    }
    DynamicReturnSpecification(const int bt_ix, const int gm_ix, const int ss_ix, const int ar_ix = InvalidIndex, const int di_ix = InvalidIndex)
        : active(true),
          arg_index_for_basic_type(bt_ix),
          arg_index_for_gridmapping(gm_ix),
          arg_index_for_subset(ss_ix),
          arg_index_for_array_size(ar_ix),
          arg_index_for_dimension(di_ix)
    {
    }
    bool activeType() const
    {
        return active &&
            (arg_index_for_basic_type != InvalidIndex
             || arg_index_for_gridmapping != InvalidIndex
             || arg_index_for_subset != InvalidIndex
             || arg_index_for_array_size != InvalidIndex);
    }
    bool activeDimension() const
    {
        return active && (arg_index_for_dimension != InvalidIndex);
    }
    bool active;
    int arg_index_for_basic_type;
    int arg_index_for_gridmapping;
    int arg_index_for_subset;
    int arg_index_for_array_size;
    int arg_index_for_dimension;
};




class FunctionType
{
public:
    FunctionType();

    /// Construct FunctionType taking no arguments.
    /// Equelle type: Function() -> returntype
    explicit FunctionType(const EquelleType& return_type);

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type);

    FunctionType(const std::vector<Variable>& args,
                 const EquelleType& return_type,
                 const Dimension& return_dimension,
                 const DynamicReturnSpecification& dynamic);

    /// Only for function with non-dynamic return types.
    EquelleType returnType() const;

    /// This version of returnType() is necessary to handle dynamic return types.
    EquelleType returnType(const std::vector<EquelleType>& argtypes) const;

    /// Explicit set the return type, for use when instantiating template functions.
    void setReturnType(const EquelleType& et);

    Dimension returnDimension(const std::vector<Dimension>& argdims) const;

    Dimension returnDimension() const;

    void setReturnDimension(const Dimension& dim);

    int dynamicSubsetReturn(const std::vector<EquelleType>& argtypes) const;

    const std::vector<Variable>& arguments() const;

    std::string equelleString() const;

private:
    std::vector<Variable> arguments_;
    EquelleType return_type_;
    Dimension return_dimension_;
    DynamicReturnSpecification dynamic_;
};




class Function
{
public:
    Function(const std::string& name);

    Function(const std::string& name, const FunctionType& type);

    void declareVariable(const std::string& name, const EquelleType& type);

    EquelleType variableType(const std::string name) const;

    Dimension variableDimension(const std::string name) const;

    bool isVariableDeclared(const std::string& name) const;

    bool isVariableAssigned(const std::string& name) const;

    void setVariableAssigned(const std::string& name, const bool assigned);

    void setVariableType(const std::string& name, const EquelleType& type);

    void setVariableDimension(const std::string& name, const Dimension& type);

    void clearLocalVariables();

    const std::set<Variable>& getLocalVariables() const;

    void setLocalVariables(const std::set<Variable>& locvars);

    const std::string& name() const;

    void setName(const std::string& name);

    const FunctionType& functionType() const;

    void setFunctionType(const FunctionType& ftype);

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const;

    void setReturnType(const EquelleType& et);

    void setTemplate(const bool is_template);

    bool isTemplate() const;

    void addInstantiation(const int index);

    const std::vector<int>& instantiations() const;

    const std::string& parentScope() const;

    void setParentScope(Function* parent_scope);

    void dump() const;

private:
    std::pair<bool, EquelleType> declared(const std::string& name) const;

    std::string name_;
    std::set<Variable> local_variables_;
    FunctionType type_;
    bool is_template_;
    std::vector<int> instantiation_indices_;
    Function* parent_scope_;
};




class SymbolTable
{
public:
    static void declareVariable(const std::string& name, const EquelleType& type);

    static void declareFunction(const std::string& name);

    static void declareFunction(const std::string& name, const FunctionType& ftype, const bool is_template = false);

    static int addFunctionInstantiation(const Function& func);

    static const Function& getFunctionInstantiation(const int index);

    static int declareNewEntitySet(const std::string& name, const int subset_entity_index);

    static bool isVariableDeclared(const std::string& name);

    static bool isVariableAssigned(const std::string& name);

    static void setVariableAssigned(const std::string& name, const bool assigned);

    static EquelleType variableType(const std::string& name);

    static void setVariableType(const std::string& name, const EquelleType& type);

    static Dimension variableDimension(const std::string& name);

    static void setVariableDimension(const std::string& name, const Dimension& dimension);

    static bool isFunctionDeclared(const std::string& name);

    static const Function& getFunction(const std::string& name);

    static Function& getMutableFunction(const std::string& name);

    static const Function& getCurrentFunction();

    static void setCurrentFunction(const std::string& name);

    static void renameCurrentFunction(const std::string& name);

    static void retypeCurrentFunction(const FunctionType& ftype);

    static void clearLocalVariablesOfCurrentFunction();

    /// Returns true if set1 is a (non-strict) subset of set2.
    static bool isSubset(const int set1, const int set2);

    static Node* program();

    static void setProgram(Node* astroot);

    static std::string equelleString(const EquelleType& type);

    static const std::string& entitySetName(const int entity_set_index);

    static int entitySetIndex(const std::string& entity_set_name);

    static BasicType entitySetType(const int entity_set_index);

    static void setEntitySetName(const int entity_set_index, const std::string& name);

    static void dump();

private:
    SymbolTable();

    ~SymbolTable();

    static SymbolTable& instance();

    void declareEntitySet(const std::string& name, const int entity_index, const int subset_entity_index);

    void declareFunctionImpl(const std::string& name, const FunctionType& ftype, const bool is_template);

    bool isFunctionDeclaredImpl(const std::string& name) const;

    void setCurrentFunctionImpl(const std::string& name);

    const Function& getFunctionImpl(const std::string& name) const;
    Function& getMutableFunctionImpl(const std::string& name);

    bool isSubsetImpl(const int set1, const int set2) const;

    void dumpImpl() const;

    std::list<Function>::iterator findFunction(const std::string& name);
    std::list<Function>::const_iterator findFunction(const std::string& name) const;

    std::vector<EntitySet>::iterator findSet(const int index);
    std::vector<EntitySet>::const_iterator findSet(const int index) const;
    std::vector<EntitySet>::const_iterator findSet(const std::string& name) const;

    int next_entityset_index_;
    std::vector<EntitySet> entitysets_;
    std::list<Function> functions_;
    std::vector<Function> function_instantiations_;
    std::list<Function>::iterator main_function_;
    std::list<Function>::iterator current_function_;
    Node* ast_root_;
};





#endif // SYMBOLTABLE_HEADER_INCLUDED
