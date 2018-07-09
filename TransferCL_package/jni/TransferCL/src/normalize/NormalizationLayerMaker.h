// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <cstring>

#include "../layer/LayerMaker.h"
#include "../TransferCLDllExport.h"

/// \brief Use to add a NormalizationLayer to a NeuralNet
///
/// A NormalizationLayer will normally be inserted as the second
/// layer in a network, after an InputLayer.  It can translate
/// and scale the input values.
PUBLICAPI
class TransferCL_EXPORT NormalizationLayerMaker : public LayerMaker2 {
public:
    float _translate;
    float _scale;
    int _batchsize;
    PUBLICAPI NormalizationLayerMaker() :
        _translate(0.0f),
        _scale(1.0f),
        _batchsize(0){
    }
    PUBLICAPI NormalizationLayerMaker *batch(int _batchsize) {
        this->_batchsize = _batchsize;
        return this;
    }
    PUBLICAPI NormalizationLayerMaker *translate(float _translate) {
        this->_translate = _translate;
        return this;
    }
    PUBLICAPI NormalizationLayerMaker *scale(float _scale) {
        this->_scale = _scale;
        return this;
    }
    PUBLICAPI static NormalizationLayerMaker *instance() {
        return new NormalizationLayerMaker();
    }
    virtual NormalizationLayerMaker *clone() const {
        return new NormalizationLayerMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

