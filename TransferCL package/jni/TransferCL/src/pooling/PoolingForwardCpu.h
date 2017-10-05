// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include "PoolingForward.h"

#define VIRTUAL virtual
#define STATIC static

class PoolingForwardCpu : public PoolingForward {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PoolingForwardCpu(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize);
    VIRTUAL void forward(int batchSize, CLWrapper *inputWrapper, CLWrapper *selectorsWrapper, CLWrapper *outputWrapper);
    VIRTUAL void forward(int batchSize, float *input, int *selectors, float *output);

    // [[[end]]]
};

