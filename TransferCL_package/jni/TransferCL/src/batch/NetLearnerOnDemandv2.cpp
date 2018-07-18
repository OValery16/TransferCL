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
#include "OnDemandBatcherv2.h"
#include "../util/stringhelper.h"
//#include "loaders/GenericLoaderv2.h"
#include "NetLearnerOnDemandv2.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI NetLearnerOnDemandv2::NetLearnerOnDemandv2(Trainer *trainer, Trainable *net, 
            GenericLoaderv2 *trainLoader, int Ntrain,
            GenericLoaderv2 *validateLoader, int Ntest,
            int fileReadBatches, int batchSize, string memory_map_file_labed,string memory_map_file_data) :
        net(net),
        learnBatcher(0),
        testBatcher(0)
//    batchSize = 128;
        {
	LOGI("1)fileBatchSize %d",fileReadBatches);
    learnAction = new NetLearnLabeledAction(trainer);

    learnBatcher = new OnDemandBatcherv2(net, learnAction, trainLoader, Ntrain, fileReadBatches, batchSize,  memory_map_file_labed, memory_map_file_data);
#if NO_POSTPROCESSING==0
	testAction = new NetForwardAction();
    testBatcher = new OnDemandBatcherv2(net, testAction, validateLoader, Ntest, fileReadBatches, batchSize, memory_map_file_labed, memory_map_file_data);
#endif
//    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    learningDone = false;
    dumpTimings = false;
}
VIRTUAL NetLearnerOnDemandv2::~NetLearnerOnDemandv2() {
#if 1//TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: ~NetLearnerOnDemandv2");
#endif


    if(learnBatcher != 0) {
        delete learnBatcher;
    }
#if NO_POSTPROCESSING==0
    if(testBatcher != 0) {
        delete testBatcher;
    }
	delete testAction;
#endif

    delete learnAction;
}
VIRTUAL void NetLearnerOnDemandv2::setSchedule(int numEpochs) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: setSchedule");
#endif


    setSchedule(numEpochs, 1);
}
VIRTUAL void NetLearnerOnDemandv2::setDumpTimings(bool dumpTimings) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: setDumpTimings");
#endif


    this->dumpTimings = dumpTimings;
}
VIRTUAL void NetLearnerOnDemandv2::setSchedule(int numEpochs, int nextEpoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: setSchedule");
#endif


    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getEpochDone");
#endif


    return learnBatcher->getEpochDone();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNextEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getNextEpoch");
#endif


    return nextEpoch;
}
//VIRTUAL void NetLearnerOnDemandv2::setLearningRate(float learningRate) {
//    this->setLearningRate(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemandv2::setLearningRate(float learningRate, float annealLearningRate) {
//    this->learningRate = learningRate;
//    this->annealLearningRate = annealLearningRate;
//}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getNextBatch");
#endif


    return learnBatcher->getNextBatch();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getNTrain() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getNTrain");
#endif


    return learnBatcher->getN();
}
PUBLICAPI VIRTUAL int NetLearnerOnDemandv2::getBatchNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getBatchNumRight");
#endif


    return learnBatcher->getNumRight();
}
PUBLICAPI VIRTUAL float NetLearnerOnDemandv2::getBatchLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: getBatchLoss");
#endif


    return learnBatcher->getLoss();
}
VIRTUAL void NetLearnerOnDemandv2::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: setBatchState");
#endif


    learnBatcher->setBatchState(nextBatch, numRight, loss);
}
PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: reset");
#endif


    timer.lap();
    learningDone = false;
    nextEpoch = 0;
    learnBatcher->reset();
#if NO_POSTPROCESSING==0
    testBatcher->reset();
#endif
}
VIRTUAL void NetLearnerOnDemandv2::postEpochTesting() {
#if 1//TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: postEpochTesting");
#endif


    cout << "dumpTimings " << dumpTimings << endl;
    if(dumpTimings) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch + 1) );
//    cout << "annealed learning rate: " << learnAction->getLearningRate()

    LOGI("DeepCL/src/batch/NetLearnerOnDemandv2.cpp: postEpochTesting: training loss: %f", learnBatcher->getLoss());
    LOGI("DeepCL/src/batch/NetLearnerOnDemandv2.cpp: postEpochTesting: train accuracy: %d/%d %f \%", learnBatcher->getNumRight() , learnBatcher->getN() , (learnBatcher->getNumRight() * 100.0f/ learnBatcher->getN()));

    //cout << " training loss: " << learnBatcher->getLoss() << endl;
    //cout << " train accuracy: " << learnBatcher->getNumRight() << "/" << learnBatcher->getN() << " " << (learnBatcher->getNumRight() * 100.0f/ learnBatcher->getN()) << "%" << std::endl;
    testBatcher->run(nextEpoch);
//    int testNumRight = batchLearnerOnDemand.test(testFilepath, fileReadBatches, batchSize, Ntest);
    cout << "test accuracy: " << testBatcher->getNumRight() << "/" << testBatcher->getN() << " " << (testBatcher->getNumRight() * 100.0f / testBatcher->getN()) << "%" << endl;
    LOGI("DeepCL/src/batch/NetLearnerOnDemandv2.cpp: test accuracy: %d/%d %f \%", testBatcher->getNumRight() , testBatcher->getN() , ((testBatcher->getNumRight() * 100.0f / testBatcher->getN())));

    timer.timeCheck("after tests");
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::tickBatch() { // means: filebatch, not low-level batch
                                               // probalby good enough for now?    
//    int epoch = nextEpoch;
//    learnAction->learningRate = learningRate * pow(annealLearningRate, epoch);
	LOGE("NetLearnerOnDemandv2::tickBatch");
    learnBatcher->tick(nextEpoch);       // returns false once all learning done (all epochs)
    if(learnBatcher->getEpochDone()) {
	#if NO_POSTPROCESSING==0
        postEpochTesting();
	#endif
        nextEpoch++;
    }
//    cout << "check learningDone nextEpoch=" << nextEpoch << " numEpochs=" << numEpochs << endl;
    if(nextEpoch == numEpochs) {
//        cout << "setting learningdone to true" << endl;
        learningDone = true;
    }
    return !learningDone;
}

PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::tickEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: tickEpoch");
#endif


//    int epoch = nextEpoch;
//    cout << "NetLearnerOnDemandv2.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << learnBatcher->getEpochDone() << endl;
//    cout << "numEpochs=" << numEpochs << endl;
    if(learnBatcher->getEpochDone()) {
        learnBatcher->reset();
    }
    while(!learnBatcher->getEpochDone()) {
        tickBatch();
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::run() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: run");
#endif


    if(learningDone) {
        reset();
    }
    while(!learningDone) {
        tickEpoch();
    }
}
PUBLICAPI VIRTUAL bool NetLearnerOnDemandv2::isLearningDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearnerOnDemandv2.cpp: isLearningDone");
#endif


    return learningDone;
}
//PUBLICAPI VIRTUAL void NetLearnerOnDemandv2::learn(float learningRate) {
//    learn(learningRate, 1.0f);
//}
//VIRTUAL void NetLearnerOnDemandv2::learn(float learningRate, float annealLearningRate) {
//    setLearningRate(learningRate, annealLearningRate);
//    run();
//}
//VIRTUAL void NetLearnerOnDemandv2::setTrainer(Trainer *trainer) {
//    this->trainer = trainer;
//}

