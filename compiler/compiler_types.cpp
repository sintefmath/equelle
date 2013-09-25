#include "compiler_types.h"


bool operator==(const VariableType& a, const VariableType& b) {
	if (a.entity_type == b.entity_type && a.collection == b.collection) {
		return true;
	}
	return false;
}

bool operator!=(const VariableType& a, const VariableType& b) {
	return !(a == b);
}