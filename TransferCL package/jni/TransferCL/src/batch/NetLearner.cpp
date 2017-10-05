// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <string>

#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/Timer.h"
#include "../net/NeuralNet.h"
#include "../net/Trainable.h"
#include "NetAction.h"
#include "../util/stringhelper.h"
#include "NetLearner.h"

#include "../dependencies.h"
#include <fstream>


using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLICAPI NetLearner::NetLearner(Trainer *trainer, Trainable *net,
        int Ntrain, float *trainData, int *trainLabels,
        int Ntest, float *testData, int *testLabels,
        int batchSize, string memory_map_file_labed,string memory_map_file_data) :
        net(net)
        {
//    annealLearningRate = 1.0f;
    numEpochs = 12;
    nextEpoch = 0;
    dumpTimings = false;
    learningDone = false;
/////////////////////
    #if 0
    FILE *file0 = fopen("/data/data/com.sony.openclexample1/preloadingData/testTrainData.raw", "wb");
    	//for (int i=0; i<10; ++i)

    	 float buffer0[2048*28*28];
    	 for (int i=0; i<2048*28*28; ++i)
    		 buffer0[i]=(float)trainData[i];
    	  //float buffer[] = { 1.0f , 1.0f , 2.0f, 1.0f , 1.0f , 9.0f};
    	  //file2 = fopen ("myfile.bin", "wb");
    	  fwrite (buffer0 , sizeof(float), sizeof(buffer0), file0);
    	  fclose (file0);
	#endif

#if 0
    FILE *file2 = fopen("/data/data/com.sony.openclexample1/preloadingData/testTrainLabelData.raw", "wb");
    	//for (int i=0; i<10; ++i)

    	 int buffer[2048];
    	 for (int i=0; i<2048; ++i)
    		 buffer[i]=(int)trainLabels[i];
    	  //float buffer[] = { 1.0f , 1.0f , 2.0f, 1.0f , 1.0f , 9.0f};
    	  //file2 = fopen ("myfile.bin", "wb");
    	  fwrite (buffer , sizeof(int), sizeof(buffer), file2);
    	  fclose (file2);
	#endif


    trainBatcher = new LearnBatcher(trainer, net, batchSize, Ntrain, trainData, trainLabels,  memory_map_file_labed, memory_map_file_data);

	#if NO_POSTPROCESSING==0
		testBatcher = new ForwardBatcher(net, batchSize, Ntest, testData, testLabels, memory_map_file_labed, memory_map_file_data);
	#endif
}
VIRTUAL NetLearner::~NetLearner() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: ~NetLearner");
#endif

    delete trainBatcher;
	#if NO_POSTPROCESSING==0
		delete testBatcher;
	#endif
}
VIRTUAL void NetLearner::setSchedule(int numEpochs) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: setSchedule");
#endif


    setSchedule(numEpochs, 0);
}
VIRTUAL void NetLearner::setDumpTimings(bool dumpTimings) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: setDumpTimings");
#endif


    this->dumpTimings = dumpTimings;
}
VIRTUAL void NetLearner::setSchedule(int numEpochs, int nextEpoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: setSchedule");
#endif


    this->numEpochs = numEpochs;
    this->nextEpoch = nextEpoch;
}
PUBLICAPI VIRTUAL void NetLearner::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: reset");
#endif


//    cout << "NetLearner::reset()" << endl;
    learningDone = false;
    nextEpoch = 0;
//    net->setTraining(true);
    trainBatcher->reset();
	#if NO_POSTPROCESSING==0
		testBatcher->reset();
	#endif
    timer.lap();
}
VIRTUAL void NetLearner::postEpochTesting() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: postEpochTesting");
#endif

   LOGI("########################");
   LOGI("####  post epoch testing #######");
   LOGI("########################");
    if(dumpTimings) {
        StatefulTimer::dump(true);
    }
//        cout << "-----------------------" << endl;
    cout << endl;
    timer.timeCheck("after epoch " + toString(nextEpoch+1));
//    cout << "annealed learning rate: " << trainBatcher->getLearningRate() <<
    LOGI( " training loss: %f", trainBatcher->getLoss());
    LOGI( " train accuracy: %d/%d %f", trainBatcher->getNumRight(), trainBatcher->getN(), (trainBatcher->getNumRight() * 100.0f/ trainBatcher->getN()));
    net->setTraining(false);
    testBatcher->run(nextEpoch);
    LOGI( "test accuracy: %d/%d ",testBatcher->getNumRight(),testBatcher->getN());
    timer.timeCheck("after tests");
}
PUBLICAPI VIRTUAL bool NetLearner::tickBatch() { // just tick one learn batch, once all done, then run testing etc
#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\batch\\NetLearner.cpp: tickbatch");
#endif
	//olivier: computing path goes here

    net->setTraining(true);
    trainBatcher->tick(nextEpoch);       // returns false once all learning done (all epochs)
    if(trainBatcher->getEpochDone()) {
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
PUBLICAPI VIRTUAL bool NetLearner::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getEpochDone");
#endif


    return trainBatcher->getEpochDone();
}
PUBLICAPI VIRTUAL int NetLearner::getNextEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getNextEpoch");
#endif


    return nextEpoch;
}
PUBLICAPI VIRTUAL int NetLearner::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getNextBatch");
#endif


    return trainBatcher->getNextBatch();
}
PUBLICAPI VIRTUAL int NetLearner::getNTrain() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getNTrain");
#endif


    return trainBatcher->getN();
}
PUBLICAPI VIRTUAL int NetLearner::getBatchNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getBatchNumRight");
#endif


    return trainBatcher->getNumRight();
}
PUBLICAPI VIRTUAL float NetLearner::getBatchLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: getBatchLoss");
#endif


    return trainBatcher->getLoss();
}
VIRTUAL void NetLearner::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: setBatchState");
#endif


    trainBatcher->setBatchState(nextBatch, numRight, loss);
//    trainBatcher->numRight = numRight;
//    trainBatcher->loss = loss;
}
PUBLICAPI VIRTUAL bool NetLearner::tickEpoch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: tickEpoch");
#endif


//    int epoch = nextEpoch;
//    cout << "NetLearner.tickEpoch epoch=" << epoch << " learningDone=" << learningDone << " epochDone=" << trainBatcher->getEpochDone() << endl;
//    cout << "numEpochs=" << numEpochs << endl;
    if(trainBatcher->getEpochDone()) {
        trainBatcher->reset();
    }
    while(!trainBatcher->getEpochDone()) {
        tickBatch();
    }
    return !learningDone;
}
PUBLICAPI VIRTUAL void NetLearner::run() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: run");
#endif


    if(learningDone) {
        reset();
    }
    while(!learningDone) {
        tickEpoch();
    }
}
PUBLICAPI VIRTUAL bool NetLearner::isLearningDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetLearner.cpp: isLearningDone");
#endif


    return learningDone;
}
//PUBLICAPI VIRTUAL void NetLearner::setLearningRate(float learningRate) {
//    this->setLearningRate(learningRate, 1.0f);
//}
//VIRTUAL void NetLearner::setLearningRate(float learningRate, float annealLearningRate) {
//    this->learningRate = learningRate;
//    this->annealLearningRate = annealLearningRate;
//}
//PUBLICAPI VIRTUAL void NetLearner::learn(float learningRate) {
//    learn(learningRate, 1.0f);
//}
//VIRTUAL void NetLearner::learn(float learningRate, float annealLearningRate) {
//    setLearningRate(learningRate, annealLearningRate);
//    run();
//}
//VIRTUAL void NetLearner::setTrainer(Trainer *trainer) {
//    this->trainer = trainer;
//}


