// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <stdexcept>

#include "TrainerStateMaker.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL bool TrainerStateMaker::created(TrainerState *state) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/TrainerStateMaker.cpp: created");
#endif


    throw runtime_error("TrainerStateMaker::created not implemented for .. this class");
}

