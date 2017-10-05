#include "CppRuntimeBoundary.h"

#include <cstdio>
#include <string>

const char *deepcl_stringToCharStar(std::string astring) {
#if DEEPCL_VERBOSE == 1
LOGI( "DeepCL/src/CppRuntimeBoundary.cpp: string astring) {");
#endif


    int len = astring.size();
    char *charStar = new char[len + 1];
    sprintf(charStar, "%s", astring.c_str());
    return charStar;
}
void deepcl_deleteCharStar(const char *charStar) {
    delete[] charStar;
}

