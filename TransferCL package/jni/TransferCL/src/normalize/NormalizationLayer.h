// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include "../layer/Layer.h"
#include "../activate/ActivationFunction.h"
#include "../util/stringhelper.h"


#define VIRTUAL virtual

class NormalizationLayerMaker;

class NormalizationLayer : public Layer, IHasToString {
public:
public:
    float translate; // apply translate first
    float scale;  // then scale
    EasyCL *const cl; // NOT owned by us
    const int outputPlanes;
    const int outputSize;
    CLWrapper *outputWrapper;
    float *output;// olivier we don t own it

    int batchSize;
    int allocatedSize;

    inline int getResultIndex(int n, int outPlane, int outRow, int outCol) const {
        return (( n
            * outputPlanes + outPlane)
            * outputSize + outRow)
            * outputSize + outCol;
    }
    inline float getResult(int n, int outPlane, int outRow, int outCol) const {
        return output[ getResultIndex(n,outPlane, outRow, outCol) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    NormalizationLayer(Layer *previousLayer, NormalizationLayerMaker *maker);
    VIRTUAL ~NormalizationLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getOutput();
    VIRTUAL ActivationFunction const *getActivationFunction();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL void persistToArray(int version, float *array);
    VIRTUAL void unpersistFromArray(int version, float const*array);
    VIRTUAL bool needsBackProp();
    VIRTUAL void printOutput() const;
    VIRTUAL void print() const;
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void forward();
    VIRTUAL void backward(float learningRate, float const *gradOutput);
    VIRTUAL int getOutputSize() const;
    VIRTUAL bool isFirstLayer() const;
    VIRTUAL float getTranslate() const;
    VIRTUAL float getScale() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL std::string toString();
    VIRTUAL std::string asString() const;

    // [[[end]]]
};

std::ostream &operator<<(std::ostream &os, NormalizationLayer &layer);
std::ostream &operator<<(std::ostream &os, NormalizationLayer const*layer);

