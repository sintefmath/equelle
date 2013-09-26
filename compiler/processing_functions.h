#pragma once


/**
  * This file contains signatures for functions that process part of the program.
  * These functions typically handle error messages, type checking etc.
  */


void yyerror(const char* s);
int yylex(void);
bool isVariableContainedInItsOwnAssignment(const char* variable_name, const char* assignment_expression);
char* getFirstUndeclaredVariableFromExpression(const char* assignment_expression);
char* getFirstUnassignedVariableFromExpression(const char* assignment_expression);
char* getFirstUndeclaredVariableFromExpressionInsideFunction(const char* assignment_expression);
char* getFirstUnassignedVariableFromExpressionInsideFunction(const char* assignment_expression);
bool areAllVariablesInsideAnExpressionDeclared(const char* assignment_expression);
bool areAllVariablesInsideAnExpressionAssigned(const char* assignment_expression);
bool areAllVariablesInsideAnExpressionInsideFunctionDeclared(const char* assignment_expression);
bool areAllVariablesInsideAnExpressionInsideFunctionAssigned(const char* assignment_expression);
bool isVariableValidAndHasSameReturnTypeAsCurrentFunction(const char* assignment_expression);
bool isLengthMismatchErrorMessageFoundInsideExpression(const char* assignment_expression);
bool isValidMappingBetweenListOfVariablesAndListOfTypes(const char* variables_list, const char* types_list);
string getAppropriateErrorIfAny(const char* expression_with_possible_errors_embedded);
VariableType getType(const char* variable_name);
int getIndexOfVariableFromGlobalCPPStructure(const char* variable_name);
int getIndexOfFunctionFromGlobalCPPStructure(const char* function_name);
GridMapping getGridMappingOfVariable(const char* variable_name);
GridMapping getGridMappingOfFunction(const char* function_name);
GridMapping getGridMappingOfVariableInsideFunction(const char* variable_name);
int getNumberOfArgumentsFromFunctionCall(const char* passed_parameter_list);
char* getFunctionNameFromStartOfDeclaration(const char* start_of_function_declaration);
VariableType getVariableType(const char* CPPType);
GridMapping getGridMapping(const char* CPPType);
string errorTypeToErrorMessage(string errorType);
string functionToAnySingularType(const char *function_name, const char *CPPType, const char *arguments_list, const string &EquelleType);
string functionToAnyCollectionType(const char *function_name, const char *CPPType, const char *arguments_list, const string &EquelleType);
string declaration_function(const char* variable_name, EntityType entity, bool collection);
string extended_plural_declaration_function(const char* variable_name, EntityType entity, const char* ON_expression, GridMapping ON_expression_grid_mapping);
string singular_assignment_function(const char* variable_name, const info* right_hand_side);
string plural_assignment_function(const char* variable_name, const info* right_hand_side);
string declaration_with_assignment_function(const char* variable_name, const info* right_hand_side);
string extended_plural_declaration_with_assignment_function(const char* variable_name, const info* rhs, const GridMapping& lhs);
string USS_assignment_function(const char* variable_name);
string USS_declaration_with_assignment_function(const char* variable_name);
string USSWD_assignment_function(const char* variable_name, const char* value);
string USSWD_declaration_with_assignment_function(const char* variable_name, const char* value);
string USCOS_assignment_function(const char* variable_name, const char* expression, GridMapping grid_mapping_of_expression);
string USCOS_declaration_with_assignment_function(const char* variable_name, const char* expression, GridMapping grid_mapping_of_expression);
string USCOS_extended_declaration_with_assignment_function(const char* variable_name, const char* expression, const char* ON_expression, GridMapping grid_mapping_of_expression, GridMapping grid_mapping_of_ON_expression);
string getCppTypeStringFromVariableType(VariableType variable);
string getEquelleTypeStringFromVariableType(VariableType variable);
bool checkIfFunctionHasAnyScalars(string& function_name);
string duplicateFunction(string& function_name);
