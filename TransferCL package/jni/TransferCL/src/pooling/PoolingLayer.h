// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include "../layer/Layer.h"
#include "../activate/ActivationLayer.h"

#define VIRTUAL virtual
#define STATIC static

class CLKernel;
class CLWrapper;
class PoolingForward;
class PoolingBackward;

class PoolingMaker;

class PoolingLayer : public Layer {
public:
    const bool padZeros;
    const int numPlanes;
    const int inputSize;
    const int poolingSize;

    const int outputSize;

    EasyCL *const cl; // NOT owned by us
    PoolingForward *poolingForwardImpl;
    PoolingBackward *poolingBackpropImpl;

    float *output;
    int *selectors;
    float *gradInput;

    CLWrapper *outputWrapper;
    CLWrapper *selectorsWrapper;
    CLWrapper *gradInputWrapper;

    CLWrapper *inputWrapper;

    bool setup;

//    bool outputCopiedToHost;
//    bool gradInputCopiedToHost;

    int batchSize;
    int allocatedSize;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PoolingLayer(EasyCL *cl, Layer *previousLayer, PoolingMaker *maker);
    VIRTUAL ~PoolingLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL int getOutputNumElements();
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL float *getGradInput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL std::string asString() const;
    int setPreviousLayer_activationLayer(const char *_activ);

    // [[[end]]]
};

