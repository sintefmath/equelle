

int main()
{
    HEAP_CHECK();
    cout << "/*" << endl << "  Copyright 2013 SINTEF ICT, Applied Mathematics." << endl << "*/" << endl;
    cout << "#include <opm/core/utility/parameters/ParameterGroup.hpp>" << endl;
    cout << "#include <opm/core/linalg/LinearSolverFactory.hpp>" << endl;
    cout << "#include <opm/core/utility/ErrorMacros.hpp>" << endl;
    cout << "#include <opm/autodiff/AutoDiffBlock.hpp>" << endl;
    cout << "#include <opm/autodiff/AutoDiffHelpers.hpp>" << endl;
    cout << "#include <opm/core/grid.h>" << endl;
    cout << "#include <opm/core/grid/GridManager.hpp>" << endl;
    cout << "#include <algorithm>" << endl;
    cout << "#include <iterator>" << endl;
    cout << "#include <iostream>" << endl;
    cout << "#include <cmath>" << endl;
    cout << endl;
    cout << "#include \"EquelleRuntimeCPU.hpp\"" << endl;
    cout << endl << endl;
    cout << "int main(int argc, char** argv)" << endl;
    cout << "{" << endl;
    cout << "Opm::parameter::ParameterGroup param(argc, argv, false);" << endl;
    cout << "EquelleRuntimeCPU er(param);" << endl;
    //cout << "UserParameters up(param, er);" << endl;
    cout << endl;
    HEAP_CHECK();
    yyparse();
	cout << "return 0;" << endl;
    cout << "}" << endl;
    HEAP_CHECK();
    return 0;
}


void yyerror(const char* s)
{
    HEAP_CHECK();
    string st = s;
    cout << st << endl;
}













// function which returns true if s2 is contained in s1
bool find1(const char* s1, const char* s2)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+()<>!=,");
    while(pch != NULL)
    {
        if(strcmp(pch, s2) == 0)
        {
            HEAP_CHECK();
            return true;
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return false;
}


// function which returns the first undeclared variable from a given expression (this function is called after the function "check1" returns false)
char* find2(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
        {
            if(strncmp(pch, "er.", 3) != 0)  // not a function
            {
                if(strcmp(pch, "wrong_type_error") != 0)    // we do this to prioritize the error checking
                {
                    bool found = false;
                    int i;
                    for(i = 0; i < varNo; i++)
                    {
                        if(strcmp(pch, var[i].name.c_str()) == 0)
                        {
                            found = true;
                            break;
                        }
                    }
                    if(found == false)
                    {
                        HEAP_CHECK();
                        return pch;
                    }
                }
            }
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return strdup("InvalidCall");
}



// function which returns the first unassigned variable from a given expression (this function is called after the function "check2" returns false)
char* find3(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        int i;
        for(i = 0; i < varNo; i++)
        {
            if(strcmp(pch, var[i].name.c_str()) == 0)
            {
                if(var[i].assigned == false)
                {
                    HEAP_CHECK();
                    return pch;
                }
                break;
            }
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return strdup("InvalidCall");
}


// function which returns the number of parameters from a given parameters list
int find4(const char *s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " ,");
    int counter = 0;
    while(pch != NULL)
    {
        counter++;
        pch = strtok (NULL, " ,");
    }
    HEAP_CHECK();
    return counter;
}


// function which returns the first undeclared variable from a given expression inside a function (this function is called after the function "check3" returns false)
char* find5(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
        {
            if(strncmp(pch, "er.", 3) != 0)  // not a function
            {
                if(strcmp(pch, "wrong_type_error") != 0)      // we do this to prioritize the error checking
                {
                    bool found = false;
                    int i;
                    for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
                        if(strcmp(pch, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0)
                        {
                            found = true;
                            break;
                        }

                        if(found == false)
                            for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
                                {
                                    found = true;
                                    break;
                                }

                                if(found == false)
                                {
                                    HEAP_CHECK();
                                    return pch;
                                }
                }
            }
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return strdup("InvalidCall");
}


// function which returns the first unassigned variable from a given expression inside a function (this function is called after the function "check4" returns false)
char* find6(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        int i;
        for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
        {
            if(strcmp(pch, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0)
            {
                if(fun[currentFunctionIndex].localVariables[i].assigned == false)
                {
                    HEAP_CHECK();
                    return pch;
                }
                break;
            }
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return strdup("InvalidCall");
}


// function which checks if each variable (one that begins with a small letter and it's not a default/user-defined function) from a given expression was declared
bool check1(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=^,");
    while(pch != NULL)
    {
        if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
        {
            if(strncmp(pch, "er.", 3) != 0 && strcmp(pch, "true") != 0 && strcmp(pch, "false") != 0 && strcmp(pch, "return") != 0)  // not a default function or a small letter keyword
            {
                bool found = false;

				//Search for global variable
				if (found == false) {
					for(int i = 0; i < varNo; i++) {
						if(strcmp(pch, var[i].name.c_str()) == 0) {
							found = true;
							break;
						}
					}
				}


				//Search for function name
				if (found == false) {
					for(int j = 0; j < funNo; j++) {
						if(strcmp(pch, fun[j].name.c_str()) == 0) {
							found = true;
							break;
						}
					}
				}

				//If we are at global scope, we have no further variables: return false
				if (found == false && insideFunction == false) {
					return false;
				}


				//Check for "header" variables in current function
				if (found == false) {
					for (int i=0; i<fun[currentFunctionIndex].noParam; ++i) {
						if (strcmp(pch, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0) {
							found = true;
							break;
						}
					}
				}


				//Check for function-local variables
				if (found == false) {
					for (int i=0; i<fun[currentFunctionIndex].noLocalVariables; ++i) {
						if (strcmp(pch, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0) {
							found = true;
							break;
						}
					}
				}

				//Variable could not be found: return false.
				if (found == false) {
					return false;
				}
            }
        }
		pch = strtok(NULL, " -+*/()<>!=^,");
    }
    HEAP_CHECK();
    return true;
}


// function which checks if each variable from a given expression was assigned to a value, and returns false if not
bool check2(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        int i;
        for(i = 0; i < varNo; i++)
        {
            if(strcmp(pch, var[i].name.c_str()) == 0)
            {
                if(var[i].assigned == false)
                {
                    HEAP_CHECK();
                    return false;
                }
                break;
            }
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return true;
}


// function which checks if each variable from a given expression (which is inside a function) is declared as a header or local variable in the current function (indicated by a global index) or outside the function
bool check3(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        if(pch[0] >= 'a' && pch[0] <= 'z')  // begins with a small letter ==> variable or function (not a number)
        {
            if(strncmp(pch, "er.", 3) != 0)  // not a default function
            {
                    int i;
                    bool taken = false;
                    for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
                        if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), pch) == 0)
                        {
                            taken = true;
                            break;
                        }
                        if(taken == false)    // not a header variable ==> local or global variable or another function call
                            for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
                                {
                                    taken = true;
                                    break;
                                }
                                if(taken == false)    // not a header or local variable ==> function call or global variable
                                {
                                    HEAP_CHECK();
                                    bool found = false;
                                    int j;
                                    for(j = 0; j < funNo; j++)
                                    {
                                        if(strcmp(pch, fun[j].name.c_str()) == 0)
                                        {
                                            found = true;
                                            break;
                                        }
                                    }
                                    if(found == false)    // not a header or local variable or another function call ==> global variable
                                    {
                                        for(int i = 0; i < varNo; i++)
                                            if(strcmp(var[i].name.c_str(), pch) == 0)
                                            {
                                                found = true;
                                                break;
                                            }
                                            if(found == false)
                                                return false;   // not a header or local or global variable or another function call ==> the variable doesn't exist
                                    }
                                }
            }
        }

        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return true;    // all the variables from the given expression are declared inside the current function
}


// function which checks if each variable from a given expression (which is inside a function) is assigned as a header or local variable in the current function (indicated by a global index) or as a global variable outside the function
bool check4(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        int i;
        bool taken = false;
        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), pch) == 0)
            {
                taken = true;     // if it's a header variable, it's already assigned
                break;
            }
            if(taken == false)
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), pch) == 0)
                    {
                        if(fun[currentFunctionIndex].localVariables[i].assigned == false)
                        {
                            HEAP_CHECK();
                            return false;
                        }
                        else
                            taken = true;
                        break;
                    }
                    if(taken == false)    // if it's not a header or local variable, it must be a global variable outside the function or another function call
                    {
                        for(int i = 0; i < varNo; i++)
                            if(strcmp(var[i].name.c_str(), pch) == 0)
                            {
                                if(var[i].assigned == false)
                                    return false;
                                else
                                    taken = true;
                                break;
                            }
                    }
                    if(taken == false)    // if it's not a header or local or global variable, it must be another function call
                    {
                        for(int i = 0; i < funNo; i++)
                            if(strcmp(fun[i].name.c_str(), pch) == 0)
                            {
                                if(fun[i].assigned == false)
                                    return false;
                                break;
                            }
                    }
                    pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return true;    // all the variables from the given expression are assigned inside the current function
}


// function which checks if the given variable corresponds to a header/local variable of the current function or to a global variable and if its type is the same as the current function's return type
bool check5(const char* s1)
{
    HEAP_CHECK();
    bool found = false;
    int i;
    for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
    {
        if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), s1) == 0)
        {
            found = true;
            break;
        }
    }
    if(found == true)
    {
        if( (fun[currentFunctionIndex].headerVariables[i].type != fun[currentFunctionIndex].type)
            || (fun[currentFunctionIndex].headerVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping
            && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY
            && fun[currentFunctionIndex].headerVariables[i].grid_mapping != GRID_MAPPING_ANY)
            )
        {
            HEAP_CHECK();
            return false;
        }
        HEAP_CHECK();
        return true;
    }

    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
        if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), s1) == 0)
        {
            found = true;
            break;
        }
        if(found == true)
        {
            if((fun[currentFunctionIndex].localVariables[i].type != fun[currentFunctionIndex].type)
                || (fun[currentFunctionIndex].localVariables[i].grid_mapping != fun[currentFunctionIndex].grid_mapping
                && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY
                && fun[currentFunctionIndex].localVariables[i].grid_mapping != GRID_MAPPING_ANY)
                )
            {
                HEAP_CHECK();
                return false;
            }
            HEAP_CHECK();
            return true;
        }

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), s1) == 0)
            {
                found = true;
                break;
            }
            if(found == true)
            {
                if((var[i].type != fun[currentFunctionIndex].type)
                    || (var[i].grid_mapping != fun[currentFunctionIndex].grid_mapping
                    && fun[currentFunctionIndex].grid_mapping != GRID_MAPPING_ANY
                    && var[i].grid_mapping != GRID_MAPPING_ANY)
                    )
                {
                    HEAP_CHECK();
                    return false;
                }
                HEAP_CHECK();
                return true;
            }

            HEAP_CHECK();
            return false;
}


// function which checks if the phrase "length_mismatch_error" is found within a string (for error checking of length mismatch operations)
bool check6(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        if(strcmp(pch, "length_mismatch_error") == 0)
        {
            HEAP_CHECK();
            return true;
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return false;    // there is no length mismatch error contained in the given expression
}

// function which checks if the phrase "wrong_type_error" is found within a string (for error checking of operations between variables)
bool check7(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,");
    while(pch != NULL)
    {
        if(strcmp(pch, "wrong_type_error") == 0)
        {
            HEAP_CHECK();
            return true;
        }
        pch = strtok (NULL, " -+*/()<>!=,");
    }
    HEAP_CHECK();
    return false;
}


// function which checks if a given array of variables corresponds to a given array of types
bool check8(const char *variable_name_list, const char *variable_type_list)
{
    HEAP_CHECK();
    char *cvariable_name_list = strdup(variable_name_list);    // we need to make a copy, because the strtok function modifies the given string
    char *cvariable_type_list = strdup(variable_type_list);    // we need to make a copy, because the strtok function modifies the given string
    char *variable_name;
    variable_name = strtok(cvariable_name_list, " ,");
    char *variable_type;
    variable_type = strtok(cvariable_type_list, " ,");
    while(variable_name != NULL && variable_type != NULL)   // they should terminate simultaneously
    {
        bool found = false;

		//Search for global variable
		if (found == false) {
			for(int i = 0; i < varNo; i++) {
				if(strcmp(variable_name, var[i].name.c_str()) == 0) {
					if (var[i].type == getVariableType(variable_type)) {
						found = true;
						break;
					}
					else {
						//Wrong type...
						return false;
					}
				}
			}
		}

		//If we are at global scope, we have no further variables: return false
		if (found == false && insideFunction == false) {
			return false;
		}

		//Check for "header" variables in current function
		if (found == false) {
			for (int i=0; i<fun[currentFunctionIndex].noParam; ++i) {
				if (strcmp(variable_name, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0) {
					if (fun[currentFunctionIndex].headerVariables[i].type == getVariableType(variable_type)) {
						found = true;
						break;
					}
					else {
						//Wrong type...
						return false;
					}
				}
			}
		}


		//Check for function-local variables
		if (found == false) {
			for (int i=0; i<fun[currentFunctionIndex].noLocalVariables; ++i) {
				if (strcmp(variable_name, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0) {
					if (fun[currentFunctionIndex].localVariables[i].type == getVariableType(variable_type)) {
						found = true;
						break;
					}
					else {
						return false;
					}
				}
			}
		}

		//Variable could not be found: return false.
		if (found == false) {
			return false;
		}

        variable_name = strtok (NULL, " ,");
        variable_type = strtok (NULL, " ,");
    }
    HEAP_CHECK();
    return true;
}


// function which checks if a given string contains any error message and, if so, it returns the appropriate message
string check9(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " -+*/()<>!=,:");
    bool errorFound = false;
    while(pch != NULL)
    {
        if(strcmp(pch, "error1") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error2") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error3") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error4") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error5") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error6") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        if(strcmp(pch, "error7") == 0)
        {
            HEAP_CHECK();
            errorFound = true;
            break;
        }
        pch = strtok (NULL, " -+*/()<>!=,:");
    }

    if(errorFound == true)
        return errorTypeToErrorMessage(pch);
    HEAP_CHECK();
    return "isOk";    // there are no error messages within the assignment (which may be caused by function calls)
}


// function which returns the type of a variable, based on its name
VariableType getType(const char* variable_name)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < varNo; i++)
    {
        if(strcmp(variable_name, var[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return var[i].type;
        }
    }
    HEAP_CHECK();
    VariableType unknown;
    unknown.entity_type = TYPE_INVALID;
    return unknown;
}


// function which returns the index of a variable, based on its name
int getIndex1(const char* s1)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < varNo; i++)
    {
        if(strcmp(s1, var[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return i;
        }
    }
    HEAP_CHECK();
    return -1;
}


// function which returns the index of a function, based on its name
int getIndex2(const char* s1)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < funNo; i++)
    {
        if(strcmp(s1, fun[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return i;
        }
    }
    HEAP_CHECK();
    return -1;
}


// function which returns the size of a variable, based on its name
GridMapping getSize1(const char* s1)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < varNo; i++)
    {
        if(strcmp(s1, var[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return var[i].grid_mapping;
        }
    }
    HEAP_CHECK();
    return GRID_MAPPING_INVALID;
}


// function which returns the return size of a function, based on its name
GridMapping getSize2(const char* s1)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < funNo; i++)
    {
        if(strcmp(s1, fun[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return fun[i].grid_mapping;
        }
    }
    HEAP_CHECK();
    return GRID_MAPPING_INVALID;
}


// function which returns the size of a header/local variable inside the current function, based on its name
GridMapping getSize3(const char* s1)
{
    HEAP_CHECK();
    int i;
    for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
    {
        if(strcmp(s1, fun[currentFunctionIndex].headerVariables[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return fun[currentFunctionIndex].headerVariables[i].grid_mapping;
        }
    }
    for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
    {
        if(strcmp(s1, fun[currentFunctionIndex].localVariables[i].name.c_str()) == 0)
        {
            HEAP_CHECK();
            return fun[currentFunctionIndex].localVariables[i].grid_mapping;
        }
    }
    HEAP_CHECK();
    return GRID_MAPPING_INVALID;
}


// function which counts the number of given arguments, separated by '@'
int getSize4(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " @");
    int ctr = 0;
    while(pch != NULL)
    {
        ctr++;
        pch = strtok (NULL, " @");
    }
    HEAP_CHECK();
    return ctr;
}


// function which receives the start of a function declaration and returns its name
char* extract(const char* s1)
{
    HEAP_CHECK();
    char *cs1 = strdup(s1);    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " =");
    HEAP_CHECK();
    return pch;
}


// function used to transfer a string within a structure to a separate memory address (of its own)
char *structureToString(const char* st)
{
    HEAP_CHECK();
    char *strA = (char*)malloc(sizeof(char)*strlen(st)+1);
    strcpy(strA, st);
    HEAP_CHECK();
    return strA;
}


// function which converts a type from C++ to its corresponding type in Equelle
VariableType getVariableType(const char* st)
{
    VariableType retval;
    retval.collection = false;
    retval.entity_type = TYPE_INVALID;

    if(strcmp(st, "Scalar") == 0) {
        retval.entity_type = TYPE_SCALAR;
        retval.collection = false;
    }
    else if(strcmp(st, "Vector") == 0) {
        retval.entity_type = TYPE_VECTOR;
        retval.collection = false;
    }
    else if(strcmp(st, "Vertex") == 0) {
        retval.entity_type =  TYPE_VERTEX;
        retval.collection = false;
    }
    else if(strcmp(st, "Edge") == 0) {
        retval.entity_type =  TYPE_EDGE;
        retval.collection = false;
    }
    else if(strcmp(st, "Face") == 0) {
        retval.entity_type =  TYPE_FACE;
        retval.collection = false;
    }
    else if(strcmp(st, "Cell") == 0) {
        retval.entity_type =  TYPE_CELL;
        retval.collection = false;
    }
    else if(strcmp(st, "ScalarAD") == 0) {
        retval.entity_type =  TYPE_SCALAR_AD;
        retval.collection = false;
    }
    else if(strcmp(st, "bool") == 0) {
        retval.entity_type =  TYPE_BOOLEAN;
        retval.collection = false;
    }
    else if(strcmp(st, "CollOfScalars") == 0) {
        retval.entity_type =  TYPE_SCALAR;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfVectors") == 0) {
        retval.entity_type =  TYPE_VECTOR;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfVertices") == 0) {
        retval.entity_type =  TYPE_VERTEX;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfEdges") == 0) {
        retval.entity_type =  TYPE_EDGE;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfFaces") == 0) {
        retval.entity_type =  TYPE_FACE;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfCells") == 0) {
        retval.entity_type =  TYPE_CELL;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfScalarsAD") == 0) {
        retval.entity_type =  TYPE_SCALAR_AD;
        retval.collection = true;
    }
    else if(strcmp(st, "CollOfBools") == 0) {
        retval.entity_type =  TYPE_BOOLEAN;
        retval.collection = true;
    }

    return retval;
}


// function which returns the corresponding size of a C++ type
GridMapping getGridMapping(const char* st)
{
    if(strcmp(st, "Scalar") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Vector") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Vertex") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Edge") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Face") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "Cell") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "ScalarAD") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "bool") == 0) {
        return GRID_MAPPING_ENTITY;
    }

    if(strcmp(st, "CollOfScalars") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfVectors") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfVertices") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfEdges") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfCells") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfScalarsAD") == 0) {
        return GRID_MAPPING_ANY;
    }

    if(strcmp(st, "CollOfBools") == 0) {
        return GRID_MAPPING_ANY;
    }

    return GRID_MAPPING_INVALID;
}



// function used to convert possible error types received when assigning a function call to a variable and converts them to explicit messages
string errorTypeToErrorMessage(string errorType)
{
    // map<string, string> myMap;

    // myMap["error1"] = "One function from the assignment is not declared";
    // myMap["error2"] = "One function from the assignment is not assigned";
    // myMap["error3"] = "The return type of a function from the assignment has been declared with a different return type";
    // myMap["error4"] = "One function from the assignment receives a different number of arguments than its signature";
    // myMap["error5"] = "One function from the assignment receives an undefined variable as an argument";
    // myMap["error6"] = "One function from the assignment receives an unassigned variable as an argument";
    // myMap["error7"] = "One function from the assignment receives arguments which do not match the function's signature";

    // return myMap(errorType);

    if(errorType == "error1")
        return "One function from the assignment is not declared";
    if(errorType == "error2")
        return "One function from the assignment is not assigned";
    if(errorType == "error3")
        return "The return type of a function from the assignment has been declared with a different return type";
    if(errorType == "error4")
        return "One function from the assignment receives a different number of arguments than its signature";
    if(errorType == "error5")
        return "One function from the assignment receives an undefined variable as an argument";
    if(errorType == "error6")
        return "One function from the assignment receives an unassigned variable as an argument";
    if(errorType == "error7")
        return "One function from the assignment receives arguments which do not match the function's signature";
    return "InvalidCall";
}


string functionToAnySingularType(const char *st1, const char *st2, const char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
        return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
        return "error2: The function is not assigned";
    }
    if(strcmp(getCppTypeStringFromVariableType(fun[getIndex2(st1)].type).c_str(), st2) != 0)
    {
        stringstream ss;
        ss << "error3: The return type of the function is not a " << st4 << " type";
        return ss.str();
    }
    if(fun[getIndex2(st1)].noParam != getSize4(st3))
    {
        return "error4: The number of arguments of the function does not correspond to the number of arguments sent";
    }
    if(check1(st3) == false)
    {
        return "error5: One input variable from the function's call is undefined";
    }
    if(check2(st3) == false)
    {
        return "error6: One input variable from the function's call is unassigned";
    }
    if(check8(st3, strdup(fun[getIndex2(st1)].paramList.c_str())) == false)
    {
        return "error7: The parameter list of the template of the function does not correspond to the given parameter list";
    }

    stringstream ss;
    ss << st1 << "(" << st3 << ")";
    return ss.str();
}



string functionToAnyCollectionType(const char *st1, const char *st2, const char *st3, const string &st4)
{
    if(getIndex2(st1) == -1)
    {
        return "error1: This function does not exist";
    }
    if(fun[getIndex2(st1)].assigned == false)
    {
        return "error2: The function is not assigned";
    }
    if(strcmp(getCppTypeStringFromVariableType(fun[getIndex2(st1)].type).c_str(), st2) != 0)
    {
        stringstream ss;
        ss << "error3: The return type of the function is not a collection of " << st4 << " type";
        return ss.str();
    }
    if(fun[getIndex2(st1)].noParam != getSize4(st3))
    {
        return "error4: The number of arguments of the function does not correspond to the number of arguments sent";
    }
    if(check1(st3) == false)
    {
        return "error5: One input variable from the function's call is undefined";
    }
    if(check2(st3) == false)
    {
        return "error6: One input variable from the function's call is unassigned";
    }
    if(check8(st3, strdup(fun[getIndex2(st1)].paramList.c_str())) == false)
    {
        return "error7: The parameter list of the template of the function does not correspond to the given parameter list";
    }

    return "ok";
}



















std::string declaration_function(const char* variable_name, EntityType entity, bool collection)
{
    HEAP_CHECK();

    string finalString;

    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
        {
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }
        }

        if(taken == true)
        {
            stringstream ss;
            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
            finalString = ss.str();
        }
        else
        {
            for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
            {
                if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                {
                    taken = true;
                    break;
                }
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                finalString = ss.str();
            }
            else
            {
                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.entity_type = entity;
                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.collection = collection;
                if (collection) {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ANY;
                }
                else {
                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
                }
                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = -1;
                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
            }
        }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                var[varNo++].name = variable_name;
                var[varNo-1].type.entity_type = entity;
                var[varNo-1].type.collection = collection;
                if (collection) {
                    var[varNo-1].grid_mapping = GRID_MAPPING_ANY;
                }
                else {
                    var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                }
                var[varNo-1].array_size = -1;
                var[varNo-1].assigned = false;
            }
    }

    HEAP_CHECK();
    return finalString.c_str();
}






string extended_plural_declaration_function(const char* variable_name, EntityType entity, const char* st3, GridMapping d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
                finalString = ss.str();
            }
            else
            {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                    {
                        taken = true;
                        break;
                    }

                    if(taken == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check7(st3) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << variable_name << "'";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check3(st3) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find5(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check4(st3) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find6(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.entity_type = entity;
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type.collection = true;
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = d1;
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = -1;
                                    fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = false;
                                }
                            }
                        }
                    }
            }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                if(check7(st3) == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << variable_name << "'";
                    finalString = ss.str();
                }
                else
                {
                    if(check1(st3) == false)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is undeclared";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check2(st3) == false)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the ON expression of the variable '" << variable_name << "' is unassigned";
                            finalString = ss.str();
                        }
                        else
                        {
                            var[varNo++].name = variable_name;
                            var[varNo-1].type.entity_type = entity;
                            var[varNo-1].grid_mapping = d1;
                            var[varNo-1].assigned = false;
                            var[varNo-1].array_size = -1;
                            var[varNo-1].type.collection = true;
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}

/**
* @param rhs Right hand side of the assignment (a = b+c => rhs is essentially "b+c")
*/
string singular_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
                finalString = ss.str();
            }
            else
            {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                    {
                        taken = true;
                        break;
                    }

                    if(taken == true)
                        if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The local variable '" << variable_name << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check9(rhs->str.c_str()) != "isOk")
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check6(rhs->str.c_str()) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(find1(rhs->str.c_str(), variable_name))
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check3(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type) << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check4(rhs->str.c_str()) == false)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type) << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(check7(rhs->str.c_str()) == true)
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the variable '" << variable_name << "'";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                    fun[currentFunctionIndex].localVariables[i].array_size = rhs->array_size;
                                                    stringstream ss;
                                                    ss << "const " << getCppTypeStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                                    finalString = ss.str();
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    else
                    {   // deduced declaration
                        if(check9(rhs->str.c_str()) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(rhs->str.c_str()) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(rhs->str.c_str(), variable_name))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(rhs->str.c_str()) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getCppTypeStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                                finalString = ss.str();
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = GRID_MAPPING_ENTITY;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = rhs->array_size;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
                if(var[i].assigned == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": The " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is reassigned";
                    finalString = ss.str();
                }
                else
                {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(rhs->str.c_str()) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            var[i].assigned = true;
                                            var[i].array_size = rhs->array_size;
                                            stringstream ss;
                                            ss << "const " << getCppTypeStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                            finalString = ss.str();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            else
            {
                // deduced declaration
                if(check9(rhs->str.c_str()) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                    finalString = ss.str();
                }
                else
                {
                    if(check6(rhs->str.c_str()) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(rhs->str.c_str(), variable_name))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check7(rhs->str.c_str()) == true)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment rhs->str of the " << getCppTypeStringFromVariableType(rhs->type)  << " variable '" << variable_name << "'";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const " << getCppTypeStringFromVariableType(rhs->type)  << " " << variable_name << " = " << rhs->str << ";";
                                        finalString = ss.str();
                                        var[varNo++].name = variable_name;
                                        var[varNo-1].type = rhs->type;
                                        var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                                        var[varNo-1].assigned = true;
                                        var[varNo-1].array_size = rhs->array_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}


string plural_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the header of the function '" << fun[currentFunctionIndex].name << "' cannot be assigned";
                finalString = ss.str();
            }
            else
            {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                    {
                        taken = true;
                        break;
                    }

                    if(taken == true)
                        if(fun[currentFunctionIndex].localVariables[i].assigned == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The local variable '" << variable_name << "' is reassigned in the function '" << fun[currentFunctionIndex].name << "'";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check9(rhs->str.c_str()) != "isOk")
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check6(rhs->str.c_str()) == true)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(find1(rhs->str.c_str(), variable_name))
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check3(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check4(rhs->str.c_str()) == false)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(check7(rhs->str.c_str()) == true)
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    if(getSize3(variable_name) != rhs->grid_mapping)
                                                        if(getSize3(variable_name) == GRID_MAPPING_ANY)
                                                        {
                                                            fun[currentFunctionIndex].localVariables[i].grid_mapping = rhs->grid_mapping;
                                                            fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                            fun[currentFunctionIndex].localVariables[i].array_size = rhs->array_size;
                                                            stringstream ss;
                                                            ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                            finalString = ss.str();
                                                        }
                                                        else
                                                        {
                                                            stringstream ss;
                                                            ss << "error at line " << currentLineNumber << ": The length of the variable '" << variable_name << "' from its definition differs than the length of its assignment in the function '" << fun[currentFunctionIndex].name << "'";
                                                            finalString = ss.str();
                                                        }
                                                    else
                                                    {
                                                        stringstream ss;
                                                        ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                        finalString = ss.str();
                                                        fun[currentFunctionIndex].localVariables[i].assigned = true;
                                                        fun[currentFunctionIndex].localVariables[i].array_size = rhs->array_size;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    else
                    {   // deduced declaration
                        if(check9(rhs->str.c_str()) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(rhs->str.c_str()) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(rhs->str.c_str(), variable_name))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(rhs->str.c_str()) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                finalString = ss.str();
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = rhs->grid_mapping;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = rhs->array_size;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
                if(var[i].assigned == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is reassigned";
                    finalString = ss.str();
                }
                else
                {
                    if(check9(rhs->str.c_str()) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(rhs->str.c_str()) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(rhs->str.c_str(), variable_name))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(rhs->str.c_str()) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(getSize1(variable_name) != rhs->grid_mapping)
                                                if(getSize1(variable_name) == GRID_MAPPING_ANY)
                                                {
                                                    var[i].grid_mapping = rhs->grid_mapping;
                                                    var[i].array_size = rhs->array_size;
                                                    var[i].assigned = true;
                                                    stringstream ss;
                                                    ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                finalString = ss.str();
                                                var[i].assigned = true;
                                                var[i].array_size = rhs->array_size;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            else
            {
                // deduced declaration
                if(check9(rhs->str.c_str()) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                    finalString = ss.str();
                }
                else
                {
                    if(check6(rhs->str.c_str()) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(rhs->str.c_str(), variable_name))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check7(rhs->str.c_str()) == true)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                        finalString = ss.str();
                                        var[varNo++].name = variable_name;
                                        var[varNo-1].type = rhs->type;
                                        var[varNo-1].grid_mapping = rhs->grid_mapping;
                                        var[varNo-1].assigned = true;
                                        var[varNo-1].array_size = rhs->array_size;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}


string declaration_with_assignment_function(const char* variable_name, const info* rhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
                finalString = ss.str();
            }
            else
            {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                    {
                        taken = true;
                        break;
                    }

                    if(taken == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '%s'" << fun[currentFunctionIndex].name << "'";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check9(rhs->str.c_str()) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(rhs->str.c_str()) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(rhs->str.c_str(), variable_name))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(rhs->str.c_str()) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                finalString = ss.str();
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = rhs->array_size;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = rhs->grid_mapping;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                if(check9(rhs->str.c_str()) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                    finalString = ss.str();
                }
                else
                {
                    if(check6(rhs->str.c_str()) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(rhs->str.c_str(), variable_name))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of the variable '" << variable_name << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check7(rhs->str.c_str()) == true)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << variable_name << "'";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                        finalString = ss.str();
                                        var[varNo++].name = variable_name;
                                        var[varNo-1].type = rhs->type;
                                        var[varNo-1].grid_mapping = rhs->grid_mapping;
                                        var[varNo-1].array_size = rhs->array_size;
                                        var[varNo-1].assigned = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}



string extended_plural_declaration_with_assignment_function(const char* variable_name, const info* rhs, const GridMapping& lhs)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        int i;
        bool taken = false;

        for(i = 0; i < fun[currentFunctionIndex].noParam; i++)
            if(strcmp(fun[currentFunctionIndex].headerVariables[i].name.c_str(), variable_name) == 0)
            {
                taken = true;
                break;
            }

            if(taken == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' exists in the header of the function '" << fun[currentFunctionIndex].name << "'";
                finalString = ss.str();
            }
            else
            {
                for(i = 0; i < fun[currentFunctionIndex].noLocalVariables; i++)
                    if(strcmp(fun[currentFunctionIndex].localVariables[i].name.c_str(), variable_name) == 0)
                    {
                        taken = true;
                        break;
                    }

                    if(taken == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared as a local variable of the function '" << fun[currentFunctionIndex].name << "'";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check9(rhs->str.c_str()) != "isOk")
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check6(rhs->str.c_str()) == true)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(find1(rhs->str.c_str(), variable_name))
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is included in its definition";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check3(rhs->str.c_str()) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find5(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check4(rhs->str.c_str()) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find6(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' from the function '" << fun[currentFunctionIndex].name << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(lhs != rhs->grid_mapping)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": The length of The variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                                finalString = ss.str();
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables++].name = variable_name;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].type = rhs->type;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].grid_mapping = lhs;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].array_size = rhs->array_size;
                                                fun[currentFunctionIndex].localVariables[fun[currentFunctionIndex].noLocalVariables-1].assigned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
            }
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                if(check9(rhs->str.c_str()) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(rhs->str.c_str());
                    finalString = ss.str();
                }
                else
                {
                    if(check6(rhs->str.c_str()) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(rhs->str.c_str(), variable_name))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(rhs->str.c_str()) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(rhs->str.c_str()) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(rhs->str.c_str()) << "' contained in the definition of The variable '" << variable_name << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(lhs != rhs->grid_mapping)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The length of The variable '" << variable_name << "' from its definition differs than the length of its assignment";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const " << getCppTypeStringFromVariableType(rhs->type) << " " << variable_name << " = " << rhs->str << ";";
                                        finalString = ss.str();
                                        var[varNo++].name = variable_name;
                                        var[varNo-1].type = rhs->type;
                                        var[varNo-1].grid_mapping = lhs;
                                        var[varNo-1].array_size = rhs->array_size;
                                        var[varNo-1].assigned = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}


















/**
* USS = User specified scalar
* a = UserSpecifiedScalar(
*/
string USS_assignment_function(const char* variable_name)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), variable_name) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
                if(var[i].assigned == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is reassigned";
                    finalString = ss.str();
                }
                else
                {
                    if(var[i].type.entity_type != TYPE_SCALAR && var[i].type.collection != false)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << variable_name << "' is declared as a " << getCppTypeStringFromVariableType(var[i].type) << " and cannot be assigned to a scalar";
                        finalString = ss.str();
                    }
                    else
                    {
                        var[i].assigned = true;
                        stringstream ss;
                        ss << "const Scalar " << variable_name << " = param.get<Scalar>(\"" << variable_name << "\");";
                        finalString = ss.str();
                    }
                }
            else
            {
                // deduced declaration
                stringstream ss;
                ss << "const Scalar " << variable_name << " = param.get<Scalar>(\"" << variable_name << "\");";
                finalString = ss.str();
                var[varNo++].name = variable_name;
                var[varNo-1].type.entity_type = TYPE_SCALAR;
                var[varNo-1].type.collection = false;
                var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                var[varNo-1].assigned = true;
                var[varNo-1].array_size = 1;
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USS_declaration_with_assignment_function(const char* st1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                stringstream ss;
                ss << "const Scalar " << st1 << " = param.get<Scalar>(\"" << st1 << "\");";
                finalString = ss.str();
                var[varNo++].name = st1;
                var[varNo-1].type.entity_type = TYPE_SCALAR;
                var[varNo-1].type.collection = false;
                var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                var[varNo-1].assigned = true;
                var[varNo-1].array_size = 1;
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_assignment_function(const char* st1, const char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar with default inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
                if(var[i].assigned == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is reassigned";
                    finalString = ss.str();
                }
                else
                {
                    if(var[i].type.entity_type != TYPE_SCALAR && var[i].type.collection != false)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is alreadt declared and cannot be assigned to a scalar";
                        finalString = ss.str();
                    }
                    else
                    {
                        var[i].assigned = true;
                        stringstream ss;
                        ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
                        finalString = ss.str();
                    }
                }
            else
            {
                // deduced declaration
                stringstream ss;
                ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
                finalString = ss.str();
                var[varNo++].name = st1;
                var[varNo-1].type.entity_type = TYPE_SCALAR;
                var[varNo-1].type.collection = false;
                var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                var[varNo-1].assigned = true;
                var[varNo-1].array_size = 1;
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USSWD_declaration_with_assignment_function(const char* st1, const char* st2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified scalar inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                stringstream ss;
                ss << "const Scalar " << st1 << " = param.getDefault(\"" << st1 << "\", " << st2 << ");";
                finalString = ss.str();
                var[varNo++].name = st1;
                var[varNo-1].type.entity_type = TYPE_SCALAR;
                var[varNo-1].type.collection = false;
                var[varNo-1].grid_mapping = GRID_MAPPING_ENTITY;
                var[varNo-1].assigned = true;
                var[varNo-1].array_size = 1;
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_assignment_function(const char* st1, const char* st2, GridMapping d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
                if(var[i].assigned == true)
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is reassigned";
                    finalString = ss.str();
                }
                else
                {
                    if(check9(st2) != "isOk")
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": " << check9(st2);
                        finalString = ss.str();
                    }
                    else
                    {
                        if(check6(st2) == true)
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(find1(st2, st1))
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check1(st2) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check2(st2) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check7(st2) == true)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(getSize1(st1) != d1)
                                                if(getSize1(st1) == GRID_MAPPING_ANY)
                                                {
                                                    var[i].grid_mapping = d1;
                                                    var[i].assigned = true;
                                                    stringstream ss;
                                                    ss << "const CollOfScalars " << st1 << " = er.getUserSpecifiedCollectionOfScalar(param, \"" << st1 << "\", " << st2 << ".size());";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                            else
                                            {
                                                stringstream ss;
                                                ss << "const CollOfScalars " << st1 << " = er.getUserSpecifiedCollectionOfScalar(param, \"" << st1 << "\", " << st2 << ".size());";
                                                finalString = ss.str();
                                                var[i].assigned = true;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            else
            {
                // deduced declaration
                if(check9(st2) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(st2);
                    finalString = ss.str();
                }
                else
                {
                    if(check6(st2) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(st2, st1))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(st2) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check7(st2) == true)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const CollOfScalars " << st1 << " = er.getUserSpecifiedCollectionOfScalar(param, \"" << st1 << "\", " << st2 << ".size());";
                                        finalString = ss.str();
                                        var[varNo++].name = st1;
                                        var[varNo-1].type.entity_type = TYPE_SCALAR;
                                        var[varNo-1].type.collection = true;
                                        var[varNo-1].grid_mapping = d1;
                                        var[varNo-1].assigned = true;
                                        var[varNo-1].array_size = -1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_declaration_with_assignment_function(const char* st1, const char* st2, GridMapping d1)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                if(check9(st2) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(st2);
                    finalString = ss.str();
                }
                else
                {
                    if(check6(st2) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(st2, st1))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(st2) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check7(st2) == true)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the assignment expression of the variable '" << st1 << "'";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        stringstream ss;
                                        ss << "const CollOfScalars " << st1 << " = er.getUserSpecifiedCollectionOfScalar(param, \"" << st1 << "\", " << st2 << ".size());";
                                        finalString = ss.str();
                                        var[varNo++].name = st1;
                                        var[varNo-1].type.entity_type = TYPE_SCALAR;
                                        var[varNo-1].type.collection = true;
                                        var[varNo-1].grid_mapping = d1;
                                        var[varNo-1].assigned = true;
                                        var[varNo-1].array_size = -1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}


string USCOS_extended_declaration_with_assignment_function(const char* st1, const char* st2, const char* st3, GridMapping d1, GridMapping d2)
{
    HEAP_CHECK();
    string finalString;
    if(insideFunction == true)
    {
        stringstream ss;
        ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' cannot be declared as a user specified collection of scalars inside a function";
        finalString = ss.str();
    }
    else
    {
        int i;
        bool declaredBefore = false;

        for(i = 0; i < varNo; i++)
            if(strcmp(var[i].name.c_str(), st1) == 0)
            {
                declaredBefore = true;
                break;
            }

            if(declaredBefore == true)
            {
                stringstream ss;
                ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is redeclared";
                finalString = ss.str();
            }
            else
            {
                if(check9(st2) != "isOk")
                {
                    stringstream ss;
                    ss << "error at line " << currentLineNumber << ": " << check9(st2);
                    finalString = ss.str();
                }
                else
                {
                    if(check6(st2) == true)
                    {
                        stringstream ss;
                        ss << "error at line " << currentLineNumber << ": Length mismatch found between two terms of an operation";
                        finalString = ss.str();
                    }
                    else
                    {
                        if(find1(st2, st1))
                        {
                            stringstream ss;
                            ss << "error at line " << currentLineNumber << ": The variable '" << st1 << "' is included in its definition";
                            finalString = ss.str();
                        }
                        else
                        {
                            if(check1(st2) == false)
                            {
                                stringstream ss;
                                ss << "error at line " << currentLineNumber << ": The variable '" << find2(st2) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                finalString = ss.str();
                            }
                            else
                            {
                                if(check2(st2) == false)
                                {
                                    stringstream ss;
                                    ss << "error at line " << currentLineNumber << ": The variable '" << find3(st2) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                    finalString = ss.str();
                                }
                                else
                                {
                                    if(check1(st3) == false)
                                    {
                                        stringstream ss;
                                        ss << "error at line " << currentLineNumber << ": The variable '" << find2(st3) << "' contained in the definition of the variable '" << st1 << "' is undeclared";
                                        finalString = ss.str();
                                    }
                                    else
                                    {
                                        if(check2(st3) == false)
                                        {
                                            stringstream ss;
                                            ss << "error at line " << currentLineNumber << ": The variable '" << find3(st3) << "' contained in the definition of the variable '" << st1 << "' is unassigned";
                                            finalString = ss.str();
                                        }
                                        else
                                        {
                                            if(check7(st3) == true)
                                            {
                                                stringstream ss;
                                                ss << "error at line " << currentLineNumber << ": There is a wrong used variable contained in the ON expression of the variable '" << st1 << "'";
                                                finalString = ss.str();
                                            }
                                            else
                                            {
                                                if(d2 != d1)
                                                {
                                                    stringstream ss;
                                                    ss << "error at line " << currentLineNumber << ": The length of the variable '" << st1 << "' from its definition differs than the length of its assignment";
                                                    finalString = ss.str();
                                                }
                                                else
                                                {
                                                    stringstream ss;
                                                    ss << "const CollOfScalars " << st1 << " = er.getUserSpecifiedCollectionOfScalar(param, \"" << st1 << "\", " << st3 << ".size());";
                                                    finalString = ss.str();
                                                    var[varNo++].name = st1;
                                                    var[varNo-1].type.entity_type = TYPE_SCALAR;
                                                    var[varNo-1].type.collection = true;
                                                    var[varNo-1].grid_mapping = d1;
                                                    var[varNo-1].assigned = true;
                                                    var[varNo-1].array_size = -1;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
    }

    HEAP_CHECK();
    return finalString;
}



string getCppTypeStringFromVariableType(VariableType v)
{
    std::stringstream ss;

    switch(v.entity_type) {
    case TYPE_SCALAR:
        ss << ((v.collection) ? "CollOfScalars" : "Scalar");
        break;
    case TYPE_SCALAR_AD:
        ss << ((v.collection) ? "CollOfScalarsAD" : "ScalarAD");
        break;
    case TYPE_VECTOR:
        ss << ((v.collection) ? "CollOfVectors" : "Vector");
        break;
    case TYPE_VERTEX:
        ss << ((v.collection) ? "CollOfVertices" : "Vertex");
        break;
    case TYPE_EDGE:
        ss << ((v.collection) ? "CollOfEdges" : "Edge");
        break;
    case TYPE_FACE:
        ss << ((v.collection) ? "CollOfFaces" : "Face");
        break;
    case TYPE_CELL:
        ss << ((v.collection) ? "CollOfCells" : "Cell");
        break;
    case TYPE_BOOLEAN:
        ss << ((v.collection) ? "CollOfBools" : "bool");
        break;
    case TYPE_INVALID:
        ss << ((v.collection) ? "CollOfInvalidTypes" : "InvalidType");
        break;
    default:
        ss << ((v.collection) ? "CollOfUnknownTypes" : "UnknownType");
        break;
    }

    return ss.str();
}


string getEquelleTypeStringFromVariableType(VariableType v)
{
    std::stringstream ss;

    switch(v.entity_type) {
    case TYPE_SCALAR:
        ss << ((v.collection) ? "scalars" : "scalar");
        break;
    case TYPE_SCALAR_AD:
        ss << ((v.collection) ? "scalarsAD" : "scalarAD");
        break;
    case TYPE_VECTOR:
        ss << ((v.collection) ? "vectors" : "vector");
        break;
    case TYPE_VERTEX:
        ss << ((v.collection) ? "vertices" : "vertex");
        break;
    case TYPE_EDGE:
        ss << ((v.collection) ? "edges" : "edge");
        break;
    case TYPE_FACE:
        ss << ((v.collection) ? "faces" : "face");
        break;
    case TYPE_CELL:
        ss << ((v.collection) ? "cells" : "cell");
        break;
    case TYPE_BOOLEAN:
        ss << ((v.collection) ? "bools" : "bool");
        break;
    case TYPE_INVALID:
        ss << ((v.collection) ? "invalid_types" : "invalid_type");
        break;
    default:
        ss << ((v.collection) ? "unknown_types" : "unknown_type");
        break;
    }

    return ss.str();
}


bool checkIfFunctionHasAnyScalars(string& st1)
{
    int z = getIndex2(st1.c_str());
    if(z == -1) {
        return false;
	}

    // we check if the function has collection of scalars in its signature
    for(int i = 0; i < fun[z].noParam; i++) {
        if(fun[z].headerVariables[i].type.entity_type == TYPE_SCALAR 
				&& fun[z].headerVariables[i].type.collection == true) {
            return true;
        }
    }

	//Check if the return type is collection of scalars
	if(fun[z].type.entity_type == TYPE_SCALAR && fun[z].type.collection == true) {
		return true;
	}
	
	return false;
}

//FIXME: THis will break down once you have a function called "a", and a variable called "astma" etc.
string duplicateFunction(string& st1)
{
    HEAP_CHECK();
    char *cs1 = strdup(st1.c_str());    // we need to make a copy, because the strtok function modifies the given string
    char *pch;
    pch = strtok(cs1, " ");     // the first found word will be "auto"
    pch = strtok (NULL, " ");   // the 2nd found word will be the function's name

    stringstream ss;
    ss << pch;
    string str = ss.str();

    if(checkIfFunctionHasAnyScalars(str))     // we create the function's brother, having ColOfScalarsAD instead of ColOfScalars everywhere
    {
        string brother = st1;
        size_t index = 0;
		
		//Replace CollOfScalars with CollOfScalarsAD
        while (true)
        {
             /* Locate the substring to replace. */
             index = brother.find("CollOfScalars", index);
             if (index == string::npos) break;

             /* Make the replacement. */
             brother.replace(index, 13, "CollOfScalarsAD");

             /* Advance index forward so the next iteration doesn't pick it up as well. */
             index += 15;
        }

		//Replace all function calls (including "self")
		for (int i=0; i<funNo; ++i) {
			string::size_type index = brother.find(fun[i].name);

			//If function call is found
			while (index != string::npos) {
				//And this function uses scalars
				if (checkIfFunctionHasAnyScalars(fun[i].name)) {
					brother.replace(index, fun[i].name.size(), fun[i].name + "AD");
				}

				index = brother.find(fun[i].name, index + fun[i].name.size());
			}
		}

        stringstream ss;
        ss << st1 << endl << endl << brother;
        return ss.str();
    }

    HEAP_CHECK();
    return st1;
}
