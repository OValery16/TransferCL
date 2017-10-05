// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <vector>
#include <boost/iostreams/device/mapped_file.hpp>

#include "NetLearnerBase.h"
#include "../net/Trainable.h"
#include "../util/Timer.h"
#include "Batcher.h"

#define VIRTUAL virtual
#define STATIC static

class NeuralNet;
//class Trainable;
class Trainer;

#include "../TransferCLDllExport.h"

/// \brief Runs multiple learning epochs using Batcher objects
///
/// Handles learning the neural net, ie running multiple epochs.
/// Uses two Batchers, one for training, one for testing, to learn 
/// the epochs.
///
/// This class expects the data to be already in memory.
/// If the data is really big, wont fit in memory, you probably
/// want to use something more like NetLearnerOnDemand, which
/// can load in a chunk of data from datafiles at a time
PUBLICAPI
class TransferCL_EXPORT NetLearner : public NetLearnerBase {
public:
    Trainable *net;
    LearnBatcher *trainBatcher;
    ForwardBatcher *testBatcher;

    /////////////////////

    /////////////////////
//    float learningRate;
//    float annealLearningRate;
//    float annealedLearningRate;

    bool dumpTimings;

    Timer timer;
    int numEpochs;
    int nextEpoch;
    bool learningDone;

//    Trainer *trainer;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    PUBLICAPI NetLearner(Trainer *trainer, Trainable *net,
    int Ntrain, float *trainData, int *trainLabels,
    int Ntest, float *testData, int *testLabels,
    int batchSize, string memory_map_file_labed,string memory_map_file_data);
    VIRTUAL ~NetLearner();
    VIRTUAL void setSchedule(int numEpochs);
    VIRTUAL void setDumpTimings(bool dumpTimings);
    VIRTUAL void setSchedule(int numEpochs, int nextEpoch);
    PUBLICAPI VIRTUAL void reset();
    VIRTUAL void postEpochTesting();
    PUBLICAPI VIRTUAL bool tickBatch();  // just tick one learn batch, once all done, then run testing etc
    PUBLICAPI VIRTUAL bool getEpochDone();
    PUBLICAPI VIRTUAL int getNextEpoch();
    PUBLICAPI VIRTUAL int getNextBatch();
    PUBLICAPI VIRTUAL int getNTrain();
    PUBLICAPI VIRTUAL int getBatchNumRight();
    PUBLICAPI VIRTUAL float getBatchLoss();
    VIRTUAL void setBatchState(int nextBatch, int numRight, float loss);
    PUBLICAPI VIRTUAL bool tickEpoch();
    PUBLICAPI VIRTUAL void run();
    PUBLICAPI VIRTUAL bool isLearningDone();

    // [[[end]]]
};


