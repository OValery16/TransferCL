// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include "../layer/LayerMaker.h"

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// creates ActivationLayers
// note: definitely NOT part of public api for now.  very unstable :-)
class TransferCL_EXPORT ActivationMaker : public LayerMaker2 {
public:
    ActivationFunction const *_activationFunction;
    ActivationMaker() :
        _activationFunction(new ReluActivation()) {
    }
    static ActivationMaker *instance() {
        return new ActivationMaker();
    }
    ActivationMaker *tanh() {
        delete this->_activationFunction;
        this->_activationFunction = new TanhActivation();
        return this;
    }
    ActivationMaker *scaledTanh() {
        delete this->_activationFunction;
        this->_activationFunction = new ScaledTanhActivation();
        return this;
    }
    ActivationMaker *relu() {
        delete this->_activationFunction;
        this->_activationFunction = new ReluActivation();
        return this;
    }
    ActivationMaker *elu() {
        delete this->_activationFunction;
        this->_activationFunction = new EluActivation();
        return this;
    }
    ActivationMaker *sigmoid() {
        delete this->_activationFunction;
        this->_activationFunction = new SigmoidActivation();
        return this;
    }
    ActivationMaker *linear() {
        delete this->_activationFunction;
        this->_activationFunction = new LinearActivation();
        return this;
    }
    ActivationMaker *fn(ActivationFunction const*_fn) {
        this->_activationFunction = _fn;
        return this;
    }
    virtual ActivationMaker *clone() const {
        return new ActivationMaker(*this); // this will copy the activationfunction pointer too
    }
    virtual Layer *createLayer(Layer *previousLayer);
};

