// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.



#pragma once

#include "../dependencies.h"
//#include "ActivationFunction.h"
#include "../conv/ConvolutionalLayer.h"
#include "../layer/Layer.h"
#include <typeinfo>

#define VIRTUAL virtual
#define STATIC static

class ActivationForward;
class ActivationBackward;
class ActivationMaker;

// this will contain only activation, and then we can factorize activations away from
// the convolutional layers etc
// thats the plan :-)
class ActivationLayer : public Layer {
public:
    const int numPlanes;
    const int inputSize;

    const int outputSize;

    ActivationFunction const *fn;

    EasyCL *const cl; // NOT owned by us
    ActivationForward *activationForwardImpl;
    ActivationBackward *activationBackpropImpl;

    float *output; // this is not guaranteed to be up to date
                // unless outputCopiedToHost is true
    float *gradInput; // this is not guaranteed to be up to date
                // unless gradInputCopiedToHost is true

    CLWrapper *outputWrapper; // this is guaranteed to be up to date
    CLWrapper *gradInputWrapper; // this is guaranteed to be up to date
    CLWrapper *selectorWrapper;

//    bool outputCopiedToHost;
//    bool gradInputCopiedToHost;

    int batchSize;
    int allocatedSize;

    bool setup;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ActivationLayer(EasyCL *cl, Layer *previousLayer, ActivationMaker *maker);
    VIRTUAL ~ActivationLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float getOutput(int n, int plane, int row, int col);
    VIRTUAL void printOutput();
    CLWrapper *getSelectorWrapper();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getOutputNumElements();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasSize() const;
    VIRTUAL float *getGradInput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL std::string asString() const;
    VIRTUAL int getPersistSize(int version) const;

    // [[[end]]]
};

