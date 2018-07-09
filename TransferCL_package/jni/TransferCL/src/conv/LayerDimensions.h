// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include "../dependencies.h"
#include <iostream>
#include <cstring>
#include <string>

using namespace std;

#include "../TransferCLDllExport.h"

inline int square(int value) {
    return value * value;
}

class TransferCL_EXPORT LayerDimensions {
public:
    int inputPlanes, inputSize, numFilters, filterSize, outputSize;
    bool padZeros, isEven;
    bool biased;
    int skip;

    int stride;
    int inputCubeSize;
    int filtersSize;
    int outputCubeSize;
    int numInputPlanes;

    int outputSizeSquared;
    int filterSizeSquared;
    int inputSizeSquared;

    int halfFilterSize;
    bool needToNormalize;
    float translate;
    float scale;
    int activationLayer;
    int previousLayer_activationLayer;

    bool useMaxPooling;
    int maxPool_spatialExtent;
    int maxPool_strides;
    int maxPool_sizeOutput;
    int batchsize;
    bool isConv;
    float momentum;
    float learning_rate;
    float weightDecay;
    bool test=0; // activate test




    LayerDimensions() {
        memset(this, 0, sizeof(LayerDimensions) );
        previousLayer_activationLayer=-1;
        test=true;
    }
    LayerDimensions(int inputPlanes, int inputSize, 
                int numFilters, int filterSize, 
                bool padZeros, bool biased) :
            inputPlanes(inputPlanes),
            inputSize(inputSize),
            numFilters(numFilters),
            filterSize(filterSize),
            padZeros(padZeros),
            biased(biased),
            activationLayer(1),
            previousLayer_activationLayer(-1)
        {
        skip = 0;
        deriveOthers();
//        std::cout << "outputSize " << outputSize << " padZeros " << padZeros << " filtersize "
//            << filterSize << " inputSize " << inputSize << std::endl;
    }

    LayerDimensions &setMomentum(float _momentum) {
    	this->momentum=_momentum;
    	return *this;
    }

    LayerDimensions &setLearningRate(float _learning_rate) {
    	this->learning_rate=_learning_rate;
    	return *this;
    }
    LayerDimensions &setWeightDecay(float _weightDecay) {
    	this->weightDecay=_weightDecay;
    	return *this;
    }

    LayerDimensions &setIsConv(bool _isConv) {
        this->isConv = _isConv;
        return *this;
    }
    LayerDimensions &setNeedToNormalize(bool _needToNormalize) {
        this->needToNormalize = _needToNormalize;
        return *this;
    }

    LayerDimensions &setStride(int _stride) {
        this->stride = _stride;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setBatchsize(int _batchsize) {
        this->batchsize = _batchsize;
        return *this;
    }
    LayerDimensions &setActivationLayer(string _activ) {

    	if (_activ == "linear")
    		this->activationLayer=1;
    	if (_activ=="relu")
    		this->activationLayer=2;
    	if (_activ=="tanh")
    		this->activationLayer=3;
    	if (_activ=="scaledtanh")
    		this->activationLayer=4;
    	if (_activ=="sigmoid")
    		this->activationLayer=5;
    	if (_activ=="elu")
    		this->activationLayer=6;
//        deriveOthers();
        return *this;
    }
    LayerDimensions &setPreviousLayer_activationLayer(const char *_activ) {
    	if (strcmp (_activ,"LINEAR")==0)
    		this->previousLayer_activationLayer=1;
    	if (strcmp (_activ,"RELU")==0)
    		this->previousLayer_activationLayer=2;
    	if (strcmp (_activ,"TANH")==0)
    		this->previousLayer_activationLayer=3;
    	if (strcmp (_activ,"SCALEDTANH")==0)
    		this->previousLayer_activationLayer=4;
    	if (strcmp (_activ,"SIGMOID")==0)
    		this->previousLayer_activationLayer=5;
    	if (strcmp (_activ,"ELU")==0)
    		this->previousLayer_activationLayer=6;
//        deriveOthers();
        return *this;
    }
    LayerDimensions &setTranslate(float _translate) {
        this->translate = _translate;
        return *this;
    }
    LayerDimensions &setScale(float _scale) {
            this->scale = _scale;
            return *this;
        }
    LayerDimensions &setMaxPool_spatialExtent(int _maxPool_spatialExtent) {
            this->maxPool_spatialExtent = _maxPool_spatialExtent;
            return *this;
        }
    LayerDimensions &setMaxPool_strides(int _maxPool_strides) {
                this->maxPool_strides = _maxPool_strides;
                deriveOthers();
                return *this;
            }
    LayerDimensions &setInputPlanes(int _planes) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumInputPlanes(int _planes) {
        this->inputPlanes = _planes;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setInputSize(int inputSize) {
        this->inputSize = inputSize;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setSkip(int skip) {
        this->skip = skip;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setNumFilters(int numFilters) {
        this->numFilters = numFilters;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setFilterSize(int filterSize) {
        this->filterSize = filterSize;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setBiased(bool biased) {
        this->biased = biased;
        deriveOthers();
        return *this;
    }
    LayerDimensions &setUseMaxPooling(bool useMaxPooling) {
            this->useMaxPooling = useMaxPooling;
            return *this;
        }
    LayerDimensions &setPadZeros(bool padZeros) {
        this->padZeros = padZeros;
        deriveOthers();
        return *this;
    }
    void deriveOthers();
    std::string buildOptionsString();
};

TransferCL_EXPORT std::ostream &operator<<(std::ostream &os, const LayerDimensions &dim);


