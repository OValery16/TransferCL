// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "NetAction.h"
#include "../net/Trainable.h"
#include "../loaders/GenericLoaderv2.h"
#include "Batcher.h"

#include "OnDemandBatcherv2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI OnDemandBatcherv2::OnDemandBatcherv2(Trainable *net, NetAction *netAction, 
            GenericLoaderv2 *loader, int N, int fileReadBatches, int batchSize, string memory_map_file_labed,string memory_map_file_data) :
            net(net),
            netAction(netAction),
            netActionBatcher(0),
            loader(loader),
            N(N),
            fileReadBatches(fileReadBatches),
            batchSize(batchSize),
            fileBatchSize(batchSize * fileReadBatches),
            inputCubeSize(net->getInputCubeSize())
        {

	//LOGI("2)fileBatchSize %d",fileBatchSize);
    numFileBatches = (N + fileBatchSize - 1) / fileBatchSize;
    dataBuffer = new float[ fileBatchSize * inputCubeSize ];
    labelsBuffer = new int[ fileBatchSize ];
    netActionBatcher = new NetActionBatcher(net, batchSize, fileBatchSize, dataBuffer, labelsBuffer, netAction,  memory_map_file_labed, memory_map_file_data);
    reset();
}
VIRTUAL OnDemandBatcherv2::~OnDemandBatcherv2() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: ~OnDemandBatcherv2");
#endif


    delete netActionBatcher;
    delete[] dataBuffer;
    delete[] labelsBuffer;
}
VIRTUAL void OnDemandBatcherv2::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: setBatchState");
#endif


    this->nextFileBatch = nextBatch / fileReadBatches;
    this->numRight = numRight;
    this->loss = loss;
    epochDone = false;
}
VIRTUAL int OnDemandBatcherv2::getBatchSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getBatchSize");
#endif


    return batchSize;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNextFileBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getNextFileBatch");
#endif


    return nextFileBatch;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getNextBatch");
#endif


    return nextFileBatch * fileReadBatches;
}
PUBLICAPI VIRTUAL float OnDemandBatcherv2::getLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getLoss");
#endif


    return loss;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getNumRight");
#endif


    return numRight;
}
PUBLICAPI VIRTUAL bool OnDemandBatcherv2::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getEpochDone");
#endif


    return epochDone;
}
PUBLICAPI VIRTUAL int OnDemandBatcherv2::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: getN");
#endif


    return N;
}
//VIRTUAL void OnDemandBatcherv2::setLearningRate(float learningRate) {
//    this->learningRate = learningRate;
//}
//VIRTUAL void OnDemandBatcherv2::setBatchSize(int batchSize) {
//    if(batchSize != this->batchSize) {
//        this->batchSize = batchSize;
////        updateBuffers();
//    }
//}
PUBLICAPI void OnDemandBatcherv2::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: reset");
#endif


//    cout << "OnDemandBatcherv2::reset()" << endl;
    numRight = 0;
    loss = 0;
    nextFileBatch = 0;
    epochDone = false;
}
PUBLICAPI bool OnDemandBatcherv2::tick(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: tick");
#endif


//    cout << "OnDemandBatcherv2::tick nextFileBatch=" << nextFileBatch << " numRight=" << numRight << 
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
    loader->load(dataBuffer, labelsBuffer, fileBatchStart, thisFileBatchSize);
    EpochResult epochResult = netActionBatcher->run(epoch);
    loss += epochResult.loss;
    numRight += epochResult.numRight;

    nextFileBatch++;
    if(nextFileBatch == numFileBatches) {
        epochDone = true;
    }
    return !epochDone;
}
PUBLICAPI EpochResult OnDemandBatcherv2::run(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/OnDemandBatcherv2.cpp: run");
#endif


//    cout << "OnDemandBatcherv2::run() epochDone=" << epochDone << endl;
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}

