/*
  Copyright 2013 SINTEF ICT, Applied Mathematics.
*/

#ifndef SYMBOLTABLE_HEADER_INCLUDED
#define SYMBOLTABLE_HEADER_INCLUDED


#include "Common.hpp"
#include "EquelleType.hpp"

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
    FunctionType();

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

    std::string equelleString() const;

private:
    std::vector<Variable> arguments_;
    EquelleType return_type_;
    DynamicReturnSpecification dynamic_;
};




class Function
{
public:
    Function(const std::string& name);

    Function(const std::string& name, const FunctionType& type);

    void declareVariable(const std::string& name, const EquelleType& type);

    EquelleType variableType(const std::string name) const;

    bool isVariableDeclared(const std::string& name) const;

    bool isVariableAssigned(const std::string& name) const;

    void setVariableAssigned(const std::string& name, const bool assigned);

    void setVariableType(const std::string& name, const EquelleType& type);

    const std::string& name() const;

    void setName(const std::string& name);

    const FunctionType& functionType() const;

    void setFunctionType(const FunctionType& ftype);

    EquelleType returnType(const std::vector<EquelleType>& argtypes) const;

private:
    std::pair<bool, EquelleType> declared(const std::string& name) const;

    std::string name_;
    std::set<Variable> local_variables_;
    FunctionType type_;
};




class SymbolTable
{
public:
    static void declareVariable(const std::string& name, const EquelleType& type);

    static void declareFunction(const std::string& name);

    static void declareFunction(const std::string& name, const FunctionType& ftype);

    static int declareNewEntitySet(const std::string& name, const int subset_entity_index);

    static bool isVariableDeclared(const std::string& name);

    static bool isVariableAssigned(const std::string& name);

    static void setVariableAssigned(const std::string& name, const bool assigned);

    static EquelleType variableType(const std::string& name);

    static void setVariableType(const std::string& name, const EquelleType& type);

    static bool isFunctionDeclared(const std::string& name);

    static const Function& getFunction(const std::string& name);

    static const Function& getCurrentFunction();

    static void setCurrentFunction(const std::string& name);

    static void renameCurrentFunction(const std::string& name);

    static void retypeCurrentFunction(const FunctionType& ftype);

    /// Returns true if set1 is a (non-strict) subset of set2.
    static bool isSubset(const int set1, const int set2);

    static Node* program();

    static void setProgram(Node* astroot);

    static std::string equelleString(const EquelleType& type);

private:
    SymbolTable();

    static SymbolTable& instance();

    void declareEntitySet(const std::string& name, const int entity_index, const int subset_entity_index);

    void declareFunctionImpl(const std::string& name, const FunctionType& ftype);

    bool isFunctionDeclaredImpl(const std::string& name) const;

    void setCurrentFunctionImpl(const std::string& name);

    const Function& getFunctionImpl(const std::string& name) const;

    bool isSubsetImpl(const int set1, const int set2) const;

    std::list<Function>::iterator findFunction(const std::string& name);
    std::list<Function>::const_iterator findFunction(const std::string& name) const;

    std::vector<EntitySet>::const_iterator findSet(const int index) const;

    int next_entityset_index_;
    std::vector<EntitySet> entitysets_;
    std::list<Function> functions_;
    std::list<Function>::iterator main_function_;
    std::list<Function>::iterator current_function_;
    Node* ast_root_;
};





#endif // SYMBOLTABLE_HEADER_INCLUDED
