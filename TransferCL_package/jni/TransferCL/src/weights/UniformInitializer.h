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

#include "WeightsInitializer.h"

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// idea of this is that it will assign random floats uniformly sampled
// in range (- multiplier / fanin) to (+ multiplier / fanin)
class TransferCL_EXPORT UniformInitializer : public WeightsInitializer {
public:
    float multiplier;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    UniformInitializer(float multiplier);
    VIRTUAL void initializeWeights(int numWeights, float *weights, int fanin);
    VIRTUAL void initializeBias(int numBias, float *bias, int fanin);

    // [[[end]]]
};

