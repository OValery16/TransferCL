// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.



#pragma once

#include "../dependencies.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "../TransferCLDllExport.h"

class LossLayerMaker;
class Layer;
class Trainer;
class TrainerMaker;

class TransferCL_EXPORT Trainable {
public:
    virtual ~Trainable() {}
    virtual int getOutputNumElements() const = 0;
    virtual float calcLoss(float const *expectedValues) = 0;
    virtual float calcLossFromLabels(int const *labels) = 0;
    virtual void setBatchSize(int batchSize) = 0;
    virtual void setTraining(bool training) = 0;
    virtual int calcNumRight(int const *labels) = 0;
    virtual void forward(float const*images) = 0;
    virtual void backwardFromLabels(int const *labels) = 0;
    virtual void backward(float const *expectedOutput) = 0;
    virtual float const *getOutput() const = 0;
    virtual LossLayerMaker *cloneLossLayerMaker() const = 0;
    virtual int getOutputPlanes() const = 0;
    virtual int getOutputSize() const = 0;
    virtual int getInputCubeSize() const = 0;
    virtual int getOutputCubeSize() const = 0;
//    virtual void setTrainer(TrainerMaker *trainer) = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:

    // [[[end]]]
};

