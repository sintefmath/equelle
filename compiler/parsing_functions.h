#pragma once

#include "compiler_types.h"

info* createFloatingPoint(const info* integral_part, const info* remainder);
info* createInteger(const info* integer);
info* createScalarsFromExpression(const info* expression);
info* createScalarsFromScalarsAndExpression(const info* scalars, const info* expression);