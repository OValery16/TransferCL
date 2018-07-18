// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include <random>
#include "../TransferCLDllExport.h"
#include "mt19937defs.h"

#ifdef _MSC_VER // apply to all msvc versions, so dont have to retest on different msvc versions
#define TR1RANDOM
#endif

#ifdef _MSC_VER // apply to all msvc versions, so dont have to retest on different msvc versions
#define NOCHRONO
#include <ctime>
#else
#include <chrono>
#endif

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// singleton version of mt19937, so we seed it once, based on current time
// and then keep getting values out of it, even if used from different places
// and classes
// probably not threadsafe
// constructor is public, so we can override it, for testing, if we want
class TransferCL_EXPORT RandomSingleton {
    private:
    // as long as myrandom stays as private, should be ok to disable the wanrings I think?
    #ifdef _WIN32
    #pragma warning(disable: 4251)
    #endif
    MT19937 myrandom;
    #ifdef _WIN32
    #pragma warning(default: 4251)
    #endif

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    RandomSingleton();
    STATIC RandomSingleton *instance();
    VIRTUAL float _uniform();
    STATIC float uniform();
    STATIC int uniformInt(int minValueInclusive, int maxValueInclusive);

    // [[[end]]]
};

