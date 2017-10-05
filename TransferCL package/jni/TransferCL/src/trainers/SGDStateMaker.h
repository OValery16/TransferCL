// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "TrainerStateMaker.h"

#define VIRTUAL virtual
#define STATIC static

class SGDStateMaker : public TrainerStateMaker {
public:
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    TrainerState *instance(EasyCL *cl, int numWeights);
    VIRTUAL bool created(TrainerState *state);

    // [[[end]]]
};

