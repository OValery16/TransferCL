// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/Timer.h"
#include "BatchLearnerOnDemand.h"
#include "../net/NeuralNet.h"
#include "../net/Trainable.h"
#include "NetAction.h"
#include "OnDemandBatcher.h"
#include "../util/stringhelper.h"
#include "NetLearnerOnDemand.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI NetLearnerOnDemand::NetLearnerOnDemand(Trainer *trainer, Trainable *net, 
            std::string trainFilepath, int Ntrain,
            std::string testFilepath, int Ntest,
            int fileReadBatches, int batchSize, string memory_map_file_labed,string memory_map_file_data) :
        net(net),
        learnBatcher(0),
        testBatcher(0)
//    batchSize = 128;
        {
    learnAction = new NetLearnLabeledAction(trainer);
    testAction = new NetForwardAction();
    learnBatcher = new OnDemandBatcher(net, learnAction, trainFilepath, Ntrain, fileReadBatches, batchSize, memory_map_file_labed, memory_map_file_data);
    testBatcher = new OnDemandBatcher(net, testAction, testFilepath, Ntest, fileReadBatches, batchSize, memory_map_file_labed, memory_map_file_data);
//    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    learningDone = false;
    dumpTimings = false;
}
VIRTUAL NetLearnerOnDemand::~NetLearnerOnDemand() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: ~NetLearnerOnDemand");
#endif


    if(learnBatcher != 0) {
        delete learnBatcher;
    }
    if(testBatcher != 0) {
        delete testBatcher;
    }
    delete testAction;
    delete learnAction;
}
VIRTUAL void NetLearnerOnDemand::setSchedule(int numEpochs) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: setSchedule");
#endif


    setSchedule(numEpochs, 1);
}
VIRTUAL void NetLearnerOnDemand::setDumpTimings(bool dumpTimings) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: setDumpTimings");
#endif


    this->dumpTimings = dumpTimings;
}
VIRTUAL void NetLearnerOnDemand::setSchedule(int numEpochs, int nextEpoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: setSchedule");
#endif


    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemand::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getEpochDone");
#endif


    return learnBatcher->getEpochDone();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemand::getNextEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getNextEpoch");
#endif


    return nextEpoch;
}
//VIRTUAL void NetLearnerOnDemand::setLearningRate(float learningRate) {
//    this->setLearningRate(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemand::setLearningRate(float learningRate, float annealLearningRate) {
//    this->learningRate = learningRate;
//    this->annealLearningRate = annealLearningRate;
//}
PUBLICAPI VIRTUAL int NetLearnerOnDemand::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getNextBatch");
#endif


    return learnBatcher->getNextBatch();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemand::getNTrain() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getNTrain");
#endif


    return learnBatcher->getN();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemand::getBatchNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getBatchNumRight");
#endif


    return learnBatcher->getNumRight();
}
PUBLICAPI VIRTUAL float NetLearnerOnDemand::getBatchLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: getBatchLoss");
#endif


    return learnBatcher->getLoss();
}
VIRTUAL void NetLearnerOnDemand::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: setBatchState");
#endif


    learnBatcher->setBatchState(nextBatch, numRight, loss);
}
PUBLICAPI VIRTUAL void NetLearnerOnDemand::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: reset");
#endif


    timer.lap();
    learningDone = false;
    nextEpoch = 0;
    learnBatcher->reset();
    testBatcher->reset();
}
VIRTUAL void NetLearnerOnDemand::postEpochTesting() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: postEpochTesting");
#endif

LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");
LOGI( "#######################################\n");

    cout << "dumpTimings " << dumpTimings << endl;
    if(dumpTimings) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch + 1) );
//    cout << "annealed learning rate: " << learnAction->getLearningRate()
    LOGI("DeepCL/src/batch/NetLearnerOnDemand.cpp: training loss: %f", learnBatcher->getLoss());
    LOGI("DeepCL/src/batch/NetLearnerOnDemand.cpp: train accuracy: %d/%d %f \%", learnBatcher->getNumRight() , learnBatcher->getN() , (learnBatcher->getNumRight() * 100.0f/ learnBatcher->getN()));
    //cout << " training loss: " << learnBatcher->getLoss() << endl;
    //cout << " train accuracy: " << learnBatcher->getNumRight() << "/" << learnBatcher->getN() << " " << (learnBatcher->getNumRight() * 100.0f/ learnBatcher->getN()) << "%" << std::endl;
    testBatcher->run(nextEpoch);
//    int testNumRight = batchLearnerOnDemand.test(testFilepath, fileReadBatches, batchSize, Ntest);
    LOGI("DeepCL/src/batch/NetLearnerOnDemand.cpp: test accuracy: %d/%d %f \%", testBatcher->getNumRight() , testBatcher->getN() , ((testBatcher->getNumRight() * 100.0f / testBatcher->getN())));

    //cout << "test accuracy: " << testBatcher->getNumRight() << "/" << testBatcher->getN() << " " << (testBatcher->getNumRight() * 100.0f / testBatcher->getN()) << "%" << endl;
    timer.timeCheck("after tests");
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemand::tickBatch() { // means: filebatch, not low-level batch
                                               // probalby good enough for now?    
//    int epoch = nextEpoch;
//    learnAction->learningRate = learningRate * pow(annealLearningRate, epoch);
    learnBatcher->tick(nextEpoch);       // returns false once all learning done (all epochs)
    if(learnBatcher->getEpochDone()) {
        postEpochTesting();
        nextEpoch++;
    }
//    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
    if(nextEpoch == numEpochs) {
//        cout << "setting learningdone to true" << endl;
        learningDone = true;
    }
    return !learningDone;
}

PUBLICAPI VIRTUAL bool NetLearnerOnDemand::tickEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: tickEpoch");
#endif


//    int epoch = nextEpoch;
//    cout << "NetLearnerOnDemand.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << learnBatcher->getEpochDone() << endl;
//    cout << "numEpochs=" << numEpochs << endl;
    if(learnBatcher->getEpochDone()) {
        learnBatcher->reset();
    }
    while(!learnBatcher->getEpochDone()) {
        tickBatch();
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL void NetLearnerOnDemand::run() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: run");
#endif


    if(learningDone) {
        reset();
    }
    while(!learningDone) {
        tickEpoch();
    }
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemand::isLearningDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemand.cpp: isLearningDone");
#endif


    return learningDone;
}
//PUBLICAPI VIRTUAL void NetLearnerOnDemand::learn(float learningRate) {
//    learn(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemand::learn(float learningRate, float annealLearningRate) {
//    setLearningRate(learningRate, annealLearningRate);
//    run();
//}
//VIRTUAL void NetLearnerOnDemand::setTrainer(Trainer *trainer) {
//    this->trainer = trainer;
//}

