#include "parsing_functions.h"

info* createFloatingPoint(const info* integral_part, const info* remainder) {
	info* retval = new info();
    retval->str = integral_part->str + "." + remainder->str;
    retval->grid_mapping = GRID_MAPPING_ENTITY;
    retval->array_size = 1;
    retval->type.entity_type =  TYPE_SCALAR;
    retval->type.collection = false;

	return retval;
}

info* createInteger(const info* integer) {
	info* retval = new info();
    retval->str = integer->str;
    retval->grid_mapping = GRID_MAPPING_ENTITY;
    retval->array_size = 1;
    retval->type.entity_type = TYPE_SCALAR;
    retval->type.collection = false;

	return retval;
}