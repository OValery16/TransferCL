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

#include "RandomSingleton.h"

#undef STATIC
#define STATIC
#undef VIRTUAL
#define VIRTUAL

PUBLIC RandomSingleton::RandomSingleton() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/RandomSingleton.cpp: RandomSingleton");
#endif


    int time = 0;
    #ifdef NOCHRONO
    {
        time_t thistime;
        ::time(&thistime);
        time = (int)thistime;
    }
    #else
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> thistime = std::chrono::high_resolution_clock::now();
        time = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds> (thistime.time_since_epoch()).count());
    }
    #endif
    srand(time);
    unsigned long seed = (rand() << 8) + rand();
    myrandom.seed(seed);
}
PUBLIC STATIC RandomSingleton *RandomSingleton::instance() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/RandomSingleton.cpp: instance");
#endif


    static RandomSingleton *thisinstance = new RandomSingleton();
    return thisinstance; // assume single-threaded, which... we are :-)
}
//    void testingonly_setInstance(RandomSingleton *testInstance) {
//        _instance = testinstance;
//    }
PUBLIC VIRTUAL float RandomSingleton::_uniform() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/RandomSingleton.cpp: _uniform");
#endif


    return myrandom() / (float)myrandom.max();
}
PUBLIC STATIC float RandomSingleton::uniform() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/RandomSingleton.cpp: uniform");
#endif


    return instance()->_uniform();
}
PUBLIC STATIC int RandomSingleton::uniformInt(int minValueInclusive, int maxValueInclusive) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/RandomSingleton.cpp: uniformInt");
#endif


    return (instance()->myrandom() % 
        (maxValueInclusive - minValueInclusive + 1) )
     + minValueInclusive;
}

