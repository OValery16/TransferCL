// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "../net/Trainable.h"
#include "../trainers/Trainer.h"
#include "Batcher2.h"
#include "NetAction2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

/// \brief constructor: pass in data to process, along with labels, network, ...
Batcher2::Batcher2(Trainable *net, NetAction2 *action,
             int batchSize, int N, 
            InputData *inputData, OutputData *outputData) :
        net(net),
        action(action),
        batchSize(batchSize),
        N(N),
        inputData(inputData),
        outputData(outputData)
            {
//    inputCubeSize = net->getInputCubeSize();
    numBatches = (N + batchSize - 1) / batchSize;
    reset();
}
VIRTUAL Batcher2::~Batcher2() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: ~Batcher2");
#endif


}
/// \brief reset to the first batch, and set epochDone to false
void Batcher2::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: reset");
#endif


    nextBatch = 0;
//    numRight = 0;
//    loss = 0;
    epochDone = false;
}
/// \brief what is the index of the next batch to process?
int Batcher2::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: getNextBatch");
#endif


    if(epochDone) {
        return 0;
    } else {
        return nextBatch;
    }
}
/// \brief for training/testing, what is error loss so far?
//VIRTUAL float Batcher2::getLoss() {
//    return loss;
//}
///// \brief for training/testing, how many right so far?
//VIRTUAL int Batcher2::getNumRight() {
//    return numRight;
//}
/// \brief how many examples in the entire set of currently loaded data?
VIRTUAL int Batcher2::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: getN");
#endif


    return N;
}
/// \brief has this epoch finished?
VIRTUAL bool Batcher2::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: getEpochDone");
#endif


    return epochDone;
}
VIRTUAL void Batcher2::setN(int N) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: setN");
#endif


    this->N = N;
    this->numBatches = (N + batchSize - 1) / batchSize;
}

/// \brief processes one single batch of data
///
/// could be learning for one batch, or prediction/testing for one batch
///
/// if most recent epoch has finished, then resets, and starts a new
/// set of learning
bool Batcher2::tick(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: tick");
#endif


//    cout << "Batcher2::tick epochDone=" << epochDone << " batch=" <<  nextBatch << endl;
//    updateVars();
    if(epochDone) {
        reset();
    }
    int batch = nextBatch;
//    std::cout << "BatchLearner.tick() batch=" << batch << std::endl;
    int batchStart = batch * batchSize;
    int thisBatchSize = batchSize;
    if(batch == numBatches - 1) {
        thisBatchSize = N - batchStart;
    }
//    std::cout << "batchSize=" << batchSize << " thisBatchSize=" << thisBatchSize << " batch=" << batch <<
//            " batchStart=" << batchStart << " data=" << (void *)data << " labels=" << labels << 
//            std::endl;
    net->setBatchSize(thisBatchSize);
    internalTick(epoch, inputData->slice(batchStart), outputData->slice(batchStart) );

//    float thisLoss = net->calcLossFromLabels(&(labels[batchStart]));
//    int thisNumRight = net->calcNumRight(&(labels[batchStart]));
//        std::cout << "thisloss " << thisLoss << " thisnumright " << thisNumRight << std::endl; 
//    loss += thisLoss;
//    numRight += thisNumRight;
    nextBatch++;
    if(nextBatch == numBatches) {
        epochDone = true;
    }
    return !epochDone;
}

VIRTUAL void Batcher2::internalTick(int epoch, InputData *inputData, OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: internalTick");
#endif


     action->run(net, epoch, nextBatch, inputData, outputData);
}

/// \brief runs batch once, for currently loaded data
///
/// could be one batch of learning, or one batch of forward propagation
/// (for test/prediction), for example
void Batcher2::run(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher2.cpp: run");
#endif


//    if(data == 0) {
//        throw runtime_error("Batcher2: no data set");
//    }
//    if(labels == 0) {
//        throw runtime_error("Batcher2: no labels set");
//    }
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
}


