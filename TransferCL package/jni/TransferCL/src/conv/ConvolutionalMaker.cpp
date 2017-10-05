// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "ConvolutionalLayer.h"

#include "ConvolutionalMaker.h"

using namespace std;

Layer *ConvolutionalMaker::createLayer(Layer *previousLayer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalMaker.cpp: createLayer");
#endif


    if(_numFilters == 0) {
        throw runtime_error("Must provide ->numFilters(numFilters)");
    }
    if(_filterSize == 0) {
        throw runtime_error("Must provide ->filterSize(filterSize)");
    }
    Layer *layer = new ConvolutionalLayer(cl, previousLayer, this);
    return layer;
}

