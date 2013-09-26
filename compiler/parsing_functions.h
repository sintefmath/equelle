#pragma once

#include "compiler_types.h"

info* createFloatingPoint(info* integral_part, info* remainder);
info* createInteger(info* integer);
info* createScalarsFromExpression(const info* expression);
info* createScalarsFromScalarsAndExpression(const info* scalars, const info* expression);