// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include "../TransferCLDllExport.h"

class Trainable;
class Trainer;
#include "../trainers/TrainingContext.h"
#include "../trainers/trainer.h"

#define VIRTUAL virtual
#define STATIC static

class TransferCL_EXPORT EpochResult {
public:
    float loss;
    int numRight;
    EpochResult(float loss, int numRight) :
        loss(loss),
        numRight(numRight) {
    }
};

class TransferCL_EXPORT NetAction {
public:
	BatchResult epochResult;
    virtual ~NetAction() {}
    virtual void run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels) = 0;
};


class TransferCL_EXPORT NetLearnLabeledAction : public NetAction {
public:
//    float learningRate;
//    float getLearningRate() {
//        return learningRate;
//    }
    Trainer *trainer;
    NetLearnLabeledAction(Trainer *trainer) :
        trainer(trainer) {
    }   
    virtual void run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels);
};


class TransferCL_EXPORT NetForwardAction : public NetAction {
public:
    NetForwardAction() {
    }
    virtual void run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels);
};


//class TransferCL_EXPORT NetBackpropAction : public NetAction {
//public:
//    float learningRate;
//    float getLearningRate() {
//        return learningRate;
//    }
//    NetBackpropAction(float learningRate) :
//        learningRate(learningRate) {
//    }
//    virtual void run(Trainable *net, float const*const batchData, int const*const batchLabels);
//};


