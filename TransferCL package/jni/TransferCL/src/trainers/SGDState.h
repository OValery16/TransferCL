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

#include "TrainerState.h"

class EasyCL;
class CLKernel;

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// Stochastic gradient descent
// learning rate, momentum, maybe annealing
// maybe each layer gets its own TrainerState object?
// at least: any layer with weights
// they could all be initialized with the same values
// but still, they each get their own object
// we could always make like a 'prototype' or 'factory'
// object that then gets passed to each weightful layer
// Maybe a 'Maker' for trainers?
class TransferCL_EXPORT SGDState : public TrainerState {
public:
    const int numWeights;

    // should store last weights
    float *lastUpdate;
    CLWrapper *lastUpdateWrapper;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~SGDState();
    SGDState(EasyCL *cl, int numWeights);

    // [[[end]]]
};

