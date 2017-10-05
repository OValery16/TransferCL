// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <cstring>

#include "../activate/ActivationFunction.h"
#include "../layer/LayerMaker.h"
#include "../weights/OriginalInitializer.h"

#include "../TransferCLDllExport.h"
#include <string>

using namespace std;

/// Use to create a convolutional layer
PUBLICAPI
class TransferCL_EXPORT ConvolutionalMaker : public LayerMaker2 {
public:
	bool _isLast;
	int _batchSize;
	int _stride;
    int _numFilters;
    int _filterSize;
    bool _padZeros;
    bool _biased;
    std::string _activationLayer;
    bool _useMaxPooling;
    int _maxPool_spatialExtent;
    int _maxPool_strides;
    WeightsInitializer *_weightsInitializer;

    PUBLICAPI ConvolutionalMaker() :
    	    _stride(0),
    	    _batchSize(0),
            _numFilters(0),
            _filterSize(0),
            _padZeros(false),
            _biased(true),
            _activationLayer("linear"),
            _weightsInitializer(new OriginalInitializer()) { // will leak slightly, but hopefully not much
    }
    PUBLICAPI ConvolutionalMaker *stride(int stride) {
        this->_stride = stride;
        return this;
    }
    PUBLICAPI static ConvolutionalMaker *instance() {
        return new ConvolutionalMaker();
    }
    ConvolutionalMaker *weightsInitializer(WeightsInitializer *weightsInitializer) {
        this->_weightsInitializer = weightsInitializer;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *numFilters(int numFilters) {
        this->_numFilters = numFilters;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *filterSize(int filterSize) {
        this->_filterSize = filterSize;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *activationLayer(string activationLayer) {
        this->_activationLayer = activationLayer;
        return this;
    }
    PUBLICAPI ConvolutionalMaker *padZeros(bool value) {
        this->_padZeros = value;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *biased() {
        this->_biased = true;
        return this;
    }    
    PUBLICAPI ConvolutionalMaker *batchSize(int batchSize) {
        this->_batchSize = batchSize;
        return this;
    }
    PUBLICAPI ConvolutionalMaker *biased(bool _biased) {
        this->_biased = _biased;
        return this;
    }

    PUBLICAPI ConvolutionalMaker *useMaxPooling(bool _useMaxPooling) {
        this->_useMaxPooling = _useMaxPooling;
        return this;
    }
    PUBLICAPI ConvolutionalMaker *maxPool_spatialExtent(int _maxPool_spatialExtent) {
        this->_maxPool_spatialExtent = _maxPool_spatialExtent;
        return this;
    }
    PUBLICAPI ConvolutionalMaker *maxPool_strides(int _maxPool_strides) {
        this->_maxPool_strides = _maxPool_strides;
        return this;
    }    
    virtual ConvolutionalMaker *clone() const {
        return new ConvolutionalMaker(*this); // this will copy the activationfunction pointer too
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

