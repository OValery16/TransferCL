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
#include <sstream>
#include <iomanip>

#include "../TransferCLDllExport.h"

class InputLayerMaker;

#define VIRTUAL virtual

class TransferCL_EXPORT InputLayer : public Layer, IHasToString {
public:
    int batchSize;
    int allocatedSize;
    EasyCL *cl; // NOT owned by us
    const int outputPlanes;
    const int outputSize;

    float const*input; // we dont own this
    float *output; // we own this :-)

    CLWrapper *outputWrapper;
    bool setup;

    inline int getOutputIndex(int n, int outPlane, int outRow, int outCol) const {
        return (( n
            * outputPlanes + outPlane)
            * outputSize + outRow)
            * outputSize + outCol;
    }
    inline float getOutput(int n, int outPlane, int outRow, int outCol) const {
        return output[ getOutputIndex(n,outPlane, outRow, outCol) ];
    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    InputLayer(InputLayerMaker *maker);
    VIRTUAL ~InputLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getOutput();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL void printOutput();
    VIRTUAL void print();
    void in(float const*images);
    VIRTUAL bool needErrorsBackprop();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void forward();
    VIRTUAL int getOutputSize() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL std::string toString();
    VIRTUAL std::string asString() const;
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();

    // [[[end]]]
};

 std::ostream &operator<<(std::ostream &os, InputLayer &layer);
 std::ostream &operator<<(std::ostream &os, InputLayer const*layer);

