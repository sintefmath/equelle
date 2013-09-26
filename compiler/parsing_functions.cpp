#include "parsing_functions.h"

info* createFloatingPoint(const info* integral_part, const info* remainder) {
    info* floating_point = new info();
    floating_point->str = integral_part->str + "." + remainder->str;
    floating_point->grid_mapping = GRID_MAPPING_ENTITY;
    floating_point->array_size = 1;
    floating_point->type.entity_type =  TYPE_SCALAR;
    floating_point->type.collection = false;

    return floating_point;
}

info* createInteger(const info* number) {
    info* integer = new info();
    integer->str = number->str;
    integer->grid_mapping = GRID_MAPPING_ENTITY;
    integer->array_size = 1;
    integer->type.entity_type = TYPE_SCALAR;
    integer->type.collection = false;

    return integer;
}

info* createScalarsFromExpression(const info* expression) {
    info* scalar = new info();

    if(expression->error_str.size() > 0) {
        scalar->error_str = expression->error_str;
        return scalar;
    }

    switch(expression->type.entity_type) {
    case TYPE_SCALAR:
        scalar->str = expression->str.c_str();
        scalar->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
        scalar->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
        scalar->type.collection = false;
		//FIXME: Why can't we simply use scalar->array_size = expression->array_size?
		if (expression->type.collection == true) {
			scalar->array_size = expression->array_size;
		}
		else {
			scalar->array_size = 1;
		}
        break;
    default:
        scalar->error_str = "The list must contain only of scalars";
        break;
    }

	return scalar;
}


info* createScalarsFromScalarsAndExpression(const info* scalars, const info* expression) {
    info* retval = new info();
	
    if(expression->error_str.size() > 0) {
        retval->error_str = expression->error_str;
        return retval;
    }
	
    switch(expression->type.entity_type) {
    case TYPE_SCALAR:
		retval->str = scalars->str + ", " + expression->str;
		retval->grid_mapping = GRID_MAPPING_INVALID;   // it mustn't have a specific grid mapping, since we won't use this structure alone
		retval->type.entity_type = TYPE_INVALID;     // it mustn't have a specific type, since we won't use this structure alone
		retval->type.collection = false;
		if (expression->type.collection == true) {
			retval->array_size = scalars->array_size + expression->array_size;
		}
		else {
			retval->array_size = scalars->array_size + 1;
		}
        break;
    default:
        retval->error_str = "The list must contain only of scalar/scalars entities";
        break;
    }

	return retval;
}
