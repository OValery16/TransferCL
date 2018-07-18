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
#include "../activate/ActivationFunction.h"
#include "../weights/OriginalInitializer.h"

#include "../TransferCLDllExport.h"
#include <string>

using namespace std;

/// \brief Use to create a fully-connected layer
PUBLICAPI
class TransferCL_EXPORT FullyConnectedMaker : public LayerMaker2 {
public:
	int _batchSize;
    int _numPlanes;
    int _imageSize;
    bool _biased;
    bool _isLast;
    WeightsInitializer *_weightsInitializer;
    string _activationLayer;

    PUBLICAPI FullyConnectedMaker() :
        _numPlanes(0),
	    _batchSize(0),
        _imageSize(0),
        _biased(true),
        _activationLayer("linear"),
        _weightsInitializer(new OriginalInitializer()) {
    }
    FullyConnectedMaker *activationLayer(string activationLayer) {
        this->_activationLayer = activationLayer;
        return this;
    }
    FullyConnectedMaker *weightsInitializer(WeightsInitializer *weightsInitializer) {
        this->_weightsInitializer = weightsInitializer;
        return this;
    }    
    PUBLICAPI FullyConnectedMaker *numPlanes(int numPlanes) {
        this->_numPlanes = numPlanes;
        return this;
    }    
    PUBLICAPI FullyConnectedMaker *batchSize(int batchSize) {
        this->_batchSize = batchSize;
        return this;
    }
    PUBLICAPI FullyConnectedMaker *isLast(bool isLastB) {
        this->_isLast = isLastB;
        return this;
    }
    PUBLICAPI FullyConnectedMaker *imageSize(int imageSize) {
        this->_imageSize = imageSize;
        return this;
    }
    PUBLICAPI FullyConnectedMaker *biased() {
        this->_biased = true;
        return this;
    }    
    FullyConnectedMaker *biased(bool _biased) {
        this->_biased = _biased;
        return this;
    }    
    PUBLICAPI static FullyConnectedMaker *instance() {
        return new FullyConnectedMaker();
    }
    virtual FullyConnectedMaker *clone() const {
        return new FullyConnectedMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};



