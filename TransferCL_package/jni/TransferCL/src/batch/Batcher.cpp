// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "NetAction.h"
#include "../trainers/Trainer.h"

#include "../dependencies.h"
#include "../net/NeuralNet.h"
#include "../loss/LossLayer.h"
#include "../loss/IAcceptsLabels.h"

#include "Batcher.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC
//#undef PUBLICAPI
//#define PUBLICAPI

/// \brief constructor: pass in data to process, along with labels, network, ...
PUBLICAPI Batcher::Batcher(Trainable *net, int batchSize, int N, float *data, int const*labels,bool trainOrTest, string memory_map_file_labed,string memory_map_file_data) :
        net(net),
        batchSize(batchSize),
        N(N),
        data(data),
        labels(labels),
        trainOrTest(trainOrTest),
        memory_map_file_labed(memory_map_file_labed),
        memory_map_file_data(memory_map_file_data)
            {
//Mapping a segment of virtual memory to the training data files.



#if MEMORY_MAP_FILE_LOADING==1
	if (trainOrTest){

		file4.open(memory_map_file_labed,boost::iostreams::mapped_file::mapmode::readonly);
		file3.open(memory_map_file_data,boost::iostreams::mapped_file::mapmode::readonly);
		if(file4.is_open()) {
			labelTest = (const int *)file4.const_data();
		} else {
			LOGI("could not map the file filename.raw");
		}
		if(file3.is_open()) {
			dataTest = (const float *)file3.const_data();
		} else {
					LOGI("could not map the file filename.raw");
		}


	}else{
		labelTest=labels;
		dataTest=data;
	}
#endif
	/////////////////////////////
    inputCubeSize = net->getInputCubeSize();
    numBatches = (N + batchSize - 1) / batchSize;
    reset();
}
VIRTUAL Batcher::~Batcher() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: ~Batcher");
#endif
file4.close();
file3.close();

}
/// \brief reset to the first batch, and set epochDone to false
PUBLICAPI void Batcher::reset() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: reset");
#endif


    nextBatch = 0;
    numRight = 0;
    loss = 0;
    epochDone = false;
}
/// \brief what is the index of the next batch to process?
PUBLICAPI int Batcher::getNextBatch() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: getNextBatch");
#endif


    if(epochDone) {
        return 0;
    } else {
        return nextBatch;
    }
}
/// \brief for training/testing, what is error loss so far?
PUBLICAPI VIRTUAL float Batcher::getLoss() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: getLoss");
#endif


    return loss;
}
/// \brief for training/testing, how many right so far?
PUBLICAPI VIRTUAL int Batcher::getNumRight() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: getNumRight");
#endif


    return numRight;
}
/// \brief how many examples in the entire set of currently loaded data?
PUBLICAPI VIRTUAL int Batcher::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: getN");
#endif


    return N;
}
/// \brief has this epoch finished?
PUBLICAPI VIRTUAL bool Batcher::getEpochDone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: getEpochDone");
#endif


    return epochDone;
}
VIRTUAL void Batcher::setBatchState(int nextBatch, int numRight, float loss) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: setBatchState");
#endif


    this->nextBatch = nextBatch;
    this->numRight = numRight;
    this->loss = loss;
}
VIRTUAL void Batcher::setN(int N) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: setN");
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
PUBLICAPI bool Batcher::tick(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: tick");
#endif


//    cout << "Batcher::tick epochDone=" << epochDone << " batch=" <<  nextBatch << endl;
//    updateVars();
#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\batch\\batcher.cpp: batcher ticker");
#endif

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
#if MEMORY_MAP_FILE_LOADING==1
    net->setBatchSize(thisBatchSize);
    internalTick(epoch, &(dataTest[ batchStart * inputCubeSize ]), &(labelTest[batchStart]));

    if (dynamic_cast<ForwardBatcher *>(this)||dynamic_cast<NetActionBatcher *>(this)){
    	net->calcLossFromLabels(&(labelTest[batchStart]/*labels[batchStart]*/));
    	net->calcNumRight(&(labelTest[batchStart]/*labels[batchStart]*/));
    	IAcceptsLabels *temp =dynamic_cast<IAcceptsLabels *>((dynamic_cast< NeuralNet * > (net))->getLastLayer());
    	temp->getLossWrapper()->copyToHost();
    	temp->getNbRightWrapper()->copyToHost();
    	CLWrapper * lossWrapper=(CLWrapper *)(temp->getLossWrapper());
    	CLWrapper * nbRightWrapper=(CLWrapper *)(temp->getNbRightWrapper());
    	float* temp0 =(float*)(lossWrapper->getHostArray());
    	int* temp1 =(int*)(nbRightWrapper->getHostArray());
		epochResult.loss=temp0[0];
		epochResult.numRight=temp1[0];


    }
#endif
#if MEMORY_MAP_FILE_LOADING==0
    net->setBatchSize(thisBatchSize);
    internalTick(epoch, &(data[ batchStart * inputCubeSize ]), &(labels[batchStart]));

    if (dynamic_cast<ForwardBatcher *>(this)||dynamic_cast<NetActionBatcher *>(this)){
    	net->calcLossFromLabels(&(labels[batchStart]));
    	net->calcNumRight(&(labels[batchStart]));
    	IAcceptsLabels *temp =dynamic_cast<IAcceptsLabels *>((dynamic_cast< NeuralNet * > (net))->getLastLayer());
    	temp->getLossWrapper()->copyToHost();
    	temp->getNbRightWrapper()->copyToHost();
    	CLWrapper * lossWrapper=(CLWrapper *)(temp->getLossWrapper());
    	CLWrapper * nbRightWrapper=(CLWrapper *)(temp->getNbRightWrapper());
    	float* temp0 =(float*)(lossWrapper->getHostArray());
    	int* temp1 =(int*)(nbRightWrapper->getHostArray());
		epochResult.loss=temp0[0];
		epochResult.numRight=temp1[0];


    }
#endif

    loss += epochResult.loss;
    numRight += epochResult.numRight;
    //LOGI("sum loss %f munRight %d",loss,numRight);
    nextBatch++;
    if(nextBatch == numBatches) {
        epochDone = true;
    }
#if TRANSFERCL_VERBOSE == 1
    LOGI("DeepCL\\src\\batch\\batcher.cpp: end batcher ticker");
#endif

    return !epochDone;
}
/// \brief runs batch once, for currently loaded data
///
/// could be one batch of learning, or one batch of forward propagation
/// (for test/prediction), for example
PUBLICAPI EpochResult Batcher::run(int epoch) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: EpochResult Batcher::run(int epoch)");
#endif


    if(data == 0) {
        throw runtime_error("Batcher: no data set");
    }
    if(labels == 0) {
        throw runtime_error("Batcher: no labels set");
    }
    if(epochDone) {
        reset();
    }
    while(!epochDone) {
    	//LOGI("EpochResult Batcher::run tick");
        tick(epoch);
    }
    EpochResult epochResult(loss, numRight);
    return epochResult;
}
LearnBatcher::LearnBatcher(Trainer *trainer, Trainable *net,
        int batchSize, int N, float *data, int const*labels, string memory_map_file_labed,string memory_map_file_data) :
    Batcher(net, batchSize, N, data, labels,true, memory_map_file_labed, memory_map_file_data),
    trainer(trainer) {
}
VIRTUAL void LearnBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: LearnBatcher::internalTick");
#endif
//olivier: computing path goes here


    TrainingContext context(epoch, nextBatch);
    epochResult=trainer->trainFromLabels(net, &context, batchData, batchLabels);
}

NetActionBatcher::NetActionBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, NetAction *netAction, string memory_map_file_labed,string memory_map_file_data) :
    Batcher(net, batchSize, N, data, labels,false, memory_map_file_labed, memory_map_file_data),
    netAction(netAction) {

	LOGI("NetActionBatcher::NetActionBatcher ERROR");
}
void NetActionBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: NetActionBatcher::internalTick");
#endif

    //olivier: computing path goes here
    netAction->run(this->net, epoch, nextBatch, batchData, batchLabels);
    epochResult=netAction->epochResult;
}
ForwardBatcher::ForwardBatcher(Trainable *net, int batchSize, int N, float *data, int const*labels, string memory_map_file_labed,string memory_map_file_data) :
    Batcher(net, batchSize, N, data, labels,false, memory_map_file_labed, memory_map_file_data) {
	LOGI( "---------------------ForwardBatcher");
}
void ForwardBatcher::internalTick(int epoch, float const*batchData, int const*batchLabels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/Batcher.cpp: internalTick");
#endif


    this->net->forward(batchData);
}

