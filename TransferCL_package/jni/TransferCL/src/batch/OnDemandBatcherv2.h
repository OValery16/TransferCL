// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

//#include "net/Trainable.h"
//#include "BatchLearner.h"

class NetActionBatcher;
class Trainable;
class NetAction;
class GenericLoaderv2;

#include "../dependencies.h"
#include "NetAction.h"

#include "../TransferCLDllExport.h"
#include <string>

using namespace std;
#define VIRTUAL virtual
#define STATIC static

/// \brief Learns an entire epoch of training, for data that wont fit in memory
///
/// This handles an entire epoch of training, loading in data in chunks
/// and then passing it to a 'Batcher' class, to train/test each chunk
///
/// If you want to run multiple epochs, you can use a 'NetLearnerOnDemand'
/// class
///
/// compared to v1, v2 recevies a GenericLoaderv2 loader object, instead of a filepath
/// so we can handle imagenet manifests etc
PUBLICAPI
class OnDemandBatcherv2 {
protected:
//    int allocatedSize;

    Trainable *net;
//    BatchLearner *batchLearner;
    NetAction *netAction; // NOt owned by us, dont delete
    NetActionBatcher *netActionBatcher;
//    std::string filepath;
    GenericLoaderv2 *loader; //NOT owned by us, dont delete
    const int N;
    const int fileReadBatches;
    const int batchSize;
    const int fileBatchSize;
    const int inputCubeSize;
    int numFileBatches;

    float *dataBuffer;
    int *labelsBuffer;

    bool epochDone;
    int numRight;
    float loss;
    int nextFileBatch;

public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI OnDemandBatcherv2(Trainable *net, NetAction *netAction,
    GenericLoaderv2 *loader, int N, int fileReadBatches, int batchSize, string memory_map_file_labed,string memory_map_file_data);
    VIRTUAL ~OnDemandBatcherv2();
    VIRTUAL void setBatchState(int nextBatch, int numRight, float loss);
    VIRTUAL int getBatchSize();
    PUBLICAPI VIRTUAL int getNextFileBatch();
    PUBLICAPI VIRTUAL int getNextBatch();
    PUBLICAPI VIRTUAL float getLoss();
    PUBLICAPI VIRTUAL int getNumRight();
    PUBLICAPI VIRTUAL bool getEpochDone();
    PUBLICAPI VIRTUAL int getN();
    PUBLICAPI void reset();
    PUBLICAPI bool tick(int epoch);
    PUBLICAPI EpochResult run(int epoch);

    // [[[end]]]
};

