// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "SGDState.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

VIRTUAL SGDState::~SGDState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGDState.cpp: ~SGDState");
#endif


    delete lastUpdateWrapper;
    delete[] lastUpdate;
}

SGDState::SGDState(EasyCL *cl, int numWeights) :
        numWeights(numWeights)
    { // should we handle bias separately?  maybe... not?
      // or each layer could have one trainer for biases, and one for the
      // non-biases?  Maybe kind of ok?

    // lastUpdate buffer never needs to change size,
    //  since number of weights is invariant with batchSize etc
    lastUpdate = new float[numWeights];
    for(int i = 0; i < numWeights; i++) {
        lastUpdate[i] = 0.0f;
    }
    lastUpdateWrapper = cl->wrap(numWeights, lastUpdate);
    lastUpdateWrapper->copyToDevice();
}

