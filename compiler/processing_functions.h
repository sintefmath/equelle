#pragma once


/**
  * This file contains signatures for functions that process part of the program.
  * These functions typically handle error messages, type checking etc.
  */


void yyerror(const char* s);
int yylex(void);
bool find1(const char* s1, const char* s2);
char* find2(const char* s1);
char* find3(const char* s1);
int find4(const char* s1);
char* find5(const char* s1);
char* find6(const char* s1);
bool check1(const char* s1);
bool check2(const char* s1);
bool check3(const char* s1);
bool check4(const char* s1);
bool check5(const char* s1);
bool check6(const char* s1);
bool check7(const char* s1);
bool check8(const char* s1, const char* s2);
string check9(const char* s1);
VariableType getType(const char* variable_name);
int getIndex1(const char* s1);
int getIndex2(const char* s1);
GridMapping getSize1(const char* s1);
GridMapping getSize2(const char* s1);
GridMapping getSize3(const char* s1);
int getSize4(const char* s1);
char* extract(const char* s1);
VariableType getVariableType(const char* st);
GridMapping getGridMapping(const char* st);
string errorTypeToErrorMessage(string errorType);
string functionToAnySingularType(const char *st1, const char *st2, const char *st3, const string &st4);
string functionToAnyCollectionType(const char *st1, const char *st2, const char *st3, const string &st4);
string declaration_function(const char* variable_name, EntityType entity, bool collection);
string extended_plural_declaration_function(const char* variable_name, EntityType entity, const char* ON_expression, GridMapping ON_expression_grid_mapping);
string singular_assignment_function(const char* variable_name, const info* right_hand_side);
string plural_assignment_function(const char* variable_name, const info* right_hand_side);
string declaration_with_assignment_function(const char* variable_name, const info* right_hand_side);
string extended_plural_declaration_with_assignment_function(const char* variable_name, const info* rhs, const GridMapping& lhs);
string USS_assignment_function(const char* st1);
string USS_declaration_with_assignment_function(const char* st1);
string USSWD_assignment_function(const char* st1, const char* st2);
string USSWD_declaration_with_assignment_function(const char* st1, const char* st2);
string USCOS_assignment_function(const char* st1, const char* st2, GridMapping d1);
string USCOS_declaration_with_assignment_function(const char* st1, const char* st2, GridMapping d1);
string USCOS_extended_declaration_with_assignment_function(const char* st1, const char* st2, const char* st3, GridMapping d1, GridMapping d2);
string getCppTypeStringFromVariableType(VariableType variable);
string getEquelleTypeStringFromVariableType(VariableType variable);
bool checkIfFunctionHasOnlyScalars(string& st1);
string duplicateFunction(string& st1);
