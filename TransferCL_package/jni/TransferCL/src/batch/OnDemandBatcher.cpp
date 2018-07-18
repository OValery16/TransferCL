// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "../loaders/GenericLoader.h"
#include "NetAction.h"
#include "../net/Trainable.h"
#include "Batcher.h"

#include "OnDemandBatcher.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI OnDemandBatcher::OnDemandBatcher(Trainable *net, NetAction *netAction, 
            std::string filepath, int N, int fileReadBatches, int batchSize, string memory_map_file_labed,string memory_map_file_data) :
            net(net),
            netAction(netAction),
            netActionBatcher(0),
            filepath(filepath),
            N(N),
            fileReadBatches(fileReadBatches),
            batchSize(batchSize),
            fileBatchSize(batchSize * fileReadBatches),
            inputCubeSize(net->getInputCubeSize())
        {
    numFileBatches = (N + fileBatchSize - 1) / fileBatchSize;
    dataBuffer = new float[ fileBatchSize * inputCubeSize ];
    labelsBuffer = new int[ fileBatchSize ];
    netActionBatcher = new NetActionBatcher(net, batchSize, fileBatchSize, dataBuffer, labelsBuffer, netAction, memory_map_file_labed, memory_map_file_data);
    reset();
}
VIRTUAL OnDemandBatcher::~OnDemandBatcher() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: ~OnDemandBatcher");
#endif


    delete netActionBatcher;
    delete[] dataBuffer;
    delete[] labelsBuffer;
}
VIRTUAL void OnDemandBatcher::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: setBatchState");
#endif


    this->nextFileBatch = nextBatch / fileReadBatches;
    this->numRight = numRight;
    this->loss = loss;
    epochDone = false;
}
VIRTUAL int OnDemandBatcher::getBatchSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getBatchSize");
#endif


    return batchSize;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNextFileBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getNextFileBatch");
#endif


    return nextFileBatch;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getNextBatch");
#endif


    return nextFileBatch * fileReadBatches;
}
PUBLICAPI VIRTUAL float OnDemandBatcher::getLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getLoss");
#endif


    return loss;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getNumRight");
#endif


    return numRight;
}
PUBLICAPI VIRTUAL bool OnDemandBatcher::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getEpochDone");
#endif


    return epochDone;
}
PUBLICAPI VIRTUAL int OnDemandBatcher::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: getN");
#endif


    return N;
}
//VIRTUAL void OnDemandBatcher::setLearningRate(float learningRate) {
//    this->learningRate = learningRate;
//}
//VIRTUAL void OnDemandBatcher::setBatchSize(int batchSize) {
//    if(batchSize != this->batchSize) {
//        this->batchSize = batchSize;
////        updateBuffers();
//    }
//}
PUBLICAPI void OnDemandBatcher::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: reset");
#endif


//    cout << "OnDemandBatcher::reset()" << endl;
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
PUBLICAPI bool OnDemandBatcher::tick(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: tick");
#endif


//    cout << "OnDemandBatcher::tick nextFileBatch=" << nextFileBatch << " numRight=" << numRight << 
//        " loss=" << loss << " epochDone=" << epochDone << endl;
//    updateBuffers();
    if(epochDone) {
        reset();
    }
    int fileBatch = nextFileBatch;
    int fileBatchStart = fileBatch * fileBatchSize;
    int thisFileBatchSize = fileBatchSize;
    if(fileBatch == numFileBatches - 1) {
        thisFileBatchSize = N - fileBatchStart;
    }
    netActionBatcher->setN(thisFileBatchSize);
//    cout << "batchlearnerondemand, read data... filebatchstart=" << fileBatchStart << " filebatchsize=" << thisFileBatchSize << endl;
    GenericLoader::load(filepath.c_str(), dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize);

    EpochResult epochResult = netActionBatcher->run(epoch);
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if(nextFileBatch == numFileBatches) {
        epochDone = true;
    }
    return !epochDone;
}
PUBLICAPI EpochResult OnDemandBatcher::run(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcher.cpp: run");
#endif


//    cout << "OnDemandBatcher::run() epochDone=" << epochDone << endl;
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}

