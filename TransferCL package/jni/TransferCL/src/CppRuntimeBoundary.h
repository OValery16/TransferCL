#pragma once

#include "dependencies.h"
#include "TransferCLDllExport.h"

#include <string>

// handles helping to call across cpp runtime boundaries

// allocates new string, returns it.  MUST call deleteCharStar to delete it
TransferCL_EXPORT const char *deepcl_stringToCharStar(std::string astring);
TransferCL_EXPORT void deepcl_deleteCharStar(const char *charStar);

