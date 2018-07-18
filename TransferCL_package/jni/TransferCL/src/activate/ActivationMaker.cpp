// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "ActivationMaker.h"
#include "ActivationLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

Layer *ActivationMaker::createLayer(Layer *previousLayer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationMaker.cpp: createLayer");
#endif


    Layer *layer = new ActivationLayer(cl, previousLayer, this);
    return layer;
}

