// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include "../fc/FullyConnectedLayer.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "Trainer.h"
#define MEASURE_TIME_PER_OPPERATION 0

class SGDState;
class CLWrapper;
class EasyCL;
class OutputData;

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// implements SGD, including momentum
// momentum defined eg in http://www.cs.toronto.edu/~gdahl/papers/momentumNesterovDeepLearning.pdf
// standard momentum:
//    dweights[t+1] = mom * dweights[t] - learningrate * gradient(weights[t])
//    weights[t+1] = weights[t] + dweights[t+1]
//
//training:
//   given weights[t], dweights[t]:
//   forward/backprop weights[t]
//   => calc dweights[t+1]
//   => calc weights[t+1]
//
class TransferCL_EXPORT SGD : public Trainer{
public:
    float momentum;
    float weightDecay;

    bool setup;

    float * testArray;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~SGD();
    VIRTUAL void setMomentum(float momentum);
    VIRTUAL void setWeightDecay(float weightDecay);
    VIRTUAL std::string asString();
    VIRTUAL void updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
    SGDState *trainerState);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData);
    VIRTUAL BatchResult trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, float const*expectedOutput);
    VIRTUAL BatchResult trainNetFromLabels(NeuralNet *net, TrainingContext *context,
    float const*input, int const*labels);
    VIRTUAL void bindState(NeuralNet *net);
    STATIC SGD *instance(EasyCL *cl, float learningRate);
    STATIC SGD *instance(EasyCL *cl, float learningRate, float momentum);
    SGD(EasyCL *cl);

    // [[[end]]]
};

