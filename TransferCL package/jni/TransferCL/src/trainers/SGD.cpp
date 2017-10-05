// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "../util/stringhelper.h"
#include "../net/NeuralNet.h"
#include "../layer/Layer.h"
#include "../loss/LossLayer.h"
#include "SGDStateMaker.h"
#include "SGDState.h"
#include "SGD.h"
#include "../loss/IAcceptsLabels.h"
#include "../batch/NetAction.h"

#if TEST_UPDATE==1
#include "../clmath/CLMathWrapper.h"
#endif

#include "../batch/BatchData.h"

using namespace std;



#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

typedef struct {
	CLWrapper * a;
	CLWrapper *b;
	float nb_a;
	int nb_b;
} s_param;



VIRTUAL SGD::~SGD() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: ~SGD");
#endif


}
VIRTUAL void SGD::setMomentum(float momentum) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: setMomentum");
#endif


    this->momentum = momentum;
}
VIRTUAL void SGD::setWeightDecay(float weightDecay) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: setWeightDecay");
#endif


    this->weightDecay = weightDecay;
}
VIRTUAL std::string SGD::asString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: string SGD::asString");
#endif


    return "SGD{ learningRate=" + toString(learningRate) + ", momentum=" + 
        toString(momentum) + " }";
}
VIRTUAL void SGD::updateWeights(CLWrapper *weightsWrapper, CLWrapper *gradWeightsWrapper,
        SGDState *trainerState) {
#if TEST_UPDATE==1
	#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/trainers/SGD.cpp: string SGD::asString");
	#endif
    int numWeights = trainerState->numWeights;
    CLWrapper *lastUpdateWrapper = trainerState->lastUpdateWrapper;
    float *gradWeightsCopy = new float[ numWeights ];
    CLWrapper *gradWeightsCopyWrapper = cl->wrap(numWeights, gradWeightsCopy);
    gradWeightsCopyWrapper->createOnDevice();

    CLMathWrapper lastUpdates_(lastUpdateWrapper);
    CLMathWrapper gradWeights_(gradWeightsWrapper);
    CLMathWrapper gradWeightsCopy_(gradWeightsCopyWrapper);
    CLMathWrapper weights_(weightsWrapper);
    LOGI("2) momentum %f learningRate %f",momentum,learningRate);

    // following all happens on gpu, via clmathwrapper:
    lastUpdates_ *= momentum;
    gradWeightsCopy_ = gradWeights_;
    gradWeightsCopy_ *= - learningRate;
    lastUpdates_ += gradWeightsCopy_;
    weights_ += lastUpdates_;
//////////////////////


    LOGI("#######################################");
    weightsWrapper->copyToHost();
    float* temp4=(float*)weightsWrapper->getHostArray();

    float er=0.0f;
    for(int i=0;i<numWeights;i++)
    	er+=abs(testArray[i]-temp4[i]);

//     for(int i=0;i<20;i++)
//     	LOGI("test[%d]=%f",i,temp4[i]);
    LOGI("error: %f",er);

    if(weightDecay > 0) {
        // apply weight decay, by multiplying the weights by (1.0f - weightDecay)
        // so weightDecay == 0 means no decay; and weightDecay == 1.0f means
        // weights go immediately to zero
        weights_ *= 1.0f - weightDecay;
    }

    delete gradWeightsCopyWrapper;
    delete[] gradWeightsCopy;

#endif
}

/* this function is run by the second thread */
void *inc_x(void *x_void_ptr0)
{
s_param *x_void_ptr=(s_param*)x_void_ptr0;
	((CLWrapper *)x_void_ptr->a)->copyToHost();
	float* array1=(float*)((CLWrapper *)x_void_ptr->a)->getHostArray();
	((CLWrapper *)x_void_ptr->b)->copyToHost();
	int* array2=(int*)((CLWrapper *)x_void_ptr->b)->getHostArray();
	x_void_ptr->nb_a=array1[0];
	x_void_ptr->nb_b=array2[0];

return NULL;

}

static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

VIRTUAL BatchResult SGD::trainNet(NeuralNet *net, TrainingContext *context,
    float const*input, OutputData *outputData) {
    	#if TRANSFERCL_VERBOSE == 1
    	LOGI( "DeepCL/src/trainers/SGD.cpp: trainNet");
    	#endif
	    // learns one batch, including updating weights
	    // doesnt have to think about running multiple batches,
	    // or loading data, or anything like that
	    bindState(net);
#if  MEASURE_TIME_PER_OPPERATION==1
	    struct timeval start, end;
	    gettimeofday(&start, NULL);

	    double elapsedSeconds1;
	    struct timespec start2;
		struct timespec end2;

			clock_gettime(CLOCK_MONOTONIC, &start2);

#endif
#if MEASURE_TIME_PER_OPPERATION==1

    	LOGI("########################");
    	LOGI("####  forward    #######");
    	LOGI("########################");
    	struct timeval start1, end1;
    	gettimeofday(&start1, NULL);

#endif
    	    net->forward(input);
#if MEASURE_TIME_PER_OPPERATION==1
    	    cl->finish();
			clock_gettime(CLOCK_MONOTONIC, &end2);
				elapsedSeconds1 = TimeSpecToSeconds(&end2) - TimeSpecToSeconds(&start2);
				LOGI("0)-----------------forward pass took %f\n\n",elapsedSeconds1*1000);
				clock_gettime(CLOCK_MONOTONIC, &start2);

    	    gettimeofday(&end1, NULL);
    	    LOGI("-----------------forward pass took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));

    	    gettimeofday(&start1, NULL);
    	    LOGI("########################");
    	    LOGI("####  loss       #######");
    	    LOGI("########################");
#endif
    	    float loss;
    	    int numRight;
    	    net->calcLoss(outputData);
//    	    float loss = net->calcLoss(outputData);
//    	    int numRight = net->calcNumRight(outputData);
//    	    LOGI(" ----------loss = %f",loss);
//    	    LOGI(" ----------numRight = %d",numRight);

#if MEASURE_TIME_PER_OPPERATION==1
    	    cl->finish();
    	    clock_gettime(CLOCK_MONOTONIC, &end2);
			elapsedSeconds1 = TimeSpecToSeconds(&end2) - TimeSpecToSeconds(&start2);
			LOGI("0)-----------------compute loss took %f\n\n",elapsedSeconds1*1000);


    	    gettimeofday(&end1, NULL);

    	    LOGI("-----------------compute loss took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));


#endif
    	    //////////////////////////

    	    CLWrapper * lossWrap=dynamic_cast<IAcceptsLabels *>(net->getLastLayer())->getLossWrapper();
    	    CLWrapper * nbWrap=dynamic_cast<IAcceptsLabels*>(net->getLastLayer())->getNbRightWrapper();

    	    s_param test;

    	    test.a=lossWrap;
    	    test.b=nbWrap;


    	    pthread_t inc_x_thread;

    	    /* create a second thread which executes inc_x(&x) */
    	    if(pthread_create(&inc_x_thread, NULL, inc_x, &test)) {

				LOGE("error pthread");

    	    }

///////////////////////////
#if MEASURE_TIME_PER_OPPERATION==1
    	    cl->finish();
    	    LOGI("########################");
    	    LOGI("####  backward   #######");
    	    LOGI("########################");
    	    clock_gettime(CLOCK_MONOTONIC, &start2);
    	    gettimeofday(&start1, NULL);
#endif
    	    if (not setup){
				for(int layerIdx = net->getNumLayers() - 2; layerIdx > 0; layerIdx--) {
					Layer *layer = net->getLayer(layerIdx);
					if(layer->needsTrainerState()) {
						if (dynamic_cast<FullyConnectedLayer*>(layer)) {
							(dynamic_cast<FullyConnectedLayer*>(layer)) ->convolutionalLayer->setMomentum(momentum);
							(dynamic_cast<FullyConnectedLayer*>(layer)) ->convolutionalLayer->setLearningRate(learningRate);
							(dynamic_cast<FullyConnectedLayer*>(layer)) ->convolutionalLayer->setWeightDecay(weightDecay);
						}else{
							layer->setMomentum(momentum);
							layer->setLearningRate(learningRate);
							layer->setWeightDecay(weightDecay);
						}
					}
				}
			setup =true;
    	    }
    	    net->backward(outputData);
#if MEASURE_TIME_PER_OPPERATION==1
    	    cl->finish();
    	    clock_gettime(CLOCK_MONOTONIC, &end2);
			elapsedSeconds1 = TimeSpecToSeconds(&end2) - TimeSpecToSeconds(&start2);
			LOGI("0)-----------------backward pass took %f\n\n",elapsedSeconds1*1000);
    	    gettimeofday(&end1, NULL);

    	    LOGI("-----------------backward pass took %f\n ms",  (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));

#endif
#if TEST_UPDATE==1
    	    //the update is merged with the backprop
    	    LOGI("########################");
    	    LOGI("####  update     #######");
    	    LOGI("########################");
    	    startTimer1=clock();
    	    int numLayers = net->getNumLayers();
    	    for(int layerIdx = numLayers - 2; layerIdx > 0; layerIdx--) {

    	        Layer *layer = net->getLayer(layerIdx);

    	        if(!layer->needsBackProp()) {
    	            break;
    	        }

    	        if(layer->needsTrainerState()) {
    	        	///////////////////////
    	        	if(dynamic_cast<ConvolutionalLayer*>(layer)){
    	        		(dynamic_cast<ConvolutionalLayer*>(layer))->testWrapper->copyToHost();
    	        		testArray=(float*)((dynamic_cast<ConvolutionalLayer*>(layer))->testWrapper->getHostArray());
    	        	}
    	        	if(dynamic_cast<FullyConnectedLayer*>(layer)){
    	        		(dynamic_cast<FullyConnectedLayer*>(layer))->convolutionalLayer->testWrapper->copyToHost();
    	        		testArray=(float*)((dynamic_cast<FullyConnectedLayer*>(layer))->convolutionalLayer->testWrapper->getHostArray());
    	        	}
    	        	///////////////////////////

    	        	LOGI("isConvLayer %d",layer->isConvLayer());
    	        	if (dynamic_cast<FullyConnectedLayer*>(layer))
    	        	    	LOGI("FullyConnectedLayer");

    	            updateWeights(layer->getWeightsWrapper(), layer->getGradWeightsWrapper(),
    	                dynamic_cast< SGDState * >(layer->getTrainerState()) );
    	            if(layer->biased()) {
    	            	////////////////////////
    	            	if(dynamic_cast<ConvolutionalLayer*>(layer)){
    	            		(dynamic_cast<ConvolutionalLayer*>(layer))->biasTestWrapper->copyToHost();
							testArray=(float*)((dynamic_cast<ConvolutionalLayer*>(layer))->biasTestWrapper->getHostArray());
						}
						if(dynamic_cast<FullyConnectedLayer*>(layer)){
							LOGI("setup");
							(dynamic_cast<FullyConnectedLayer*>(layer))->convolutionalLayer->biasTestWrapper->copyToHost();
							testArray=(float*)((dynamic_cast<FullyConnectedLayer*>(layer))->convolutionalLayer->biasTestWrapper->getHostArray());
						}
						/////////////////////

    	                updateWeights(layer->getBiasWrapper(), layer->getGradBiasWrapper(),
    	                    dynamic_cast< SGDState * >(layer->getBiasTrainerState()) );
    	            }
    	        }
    	    }
    	    stopTimer1=clock();
    	    LOGI("update pass took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

    	    LOGI("########################");
    	    LOGI("####  end update #######");
    	    LOGI("########################");
#endif


    	    if(pthread_join(inc_x_thread, NULL)) {

    	    	LOGE("error pthread");


    	    }
    	    loss=test.nb_a;
    	    numRight=test.nb_b;
			#if DISPLAY_LOSS ==1
						LOGI("loss=%f numRight=%d",loss,numRight);
			#endif


#if 0//MEASURE_TIME_PER_OPPERATION==0
    gettimeofday(&end, NULL);

    LOGI("-----------------one iteration took took %f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));

//    stopTimer1=clock();
//    LOGI("-----------------one iteration took took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
#endif
    //LOGI("###############################################################\n\n\n");

    	    return BatchResult(loss, numRight);
    	}

VIRTUAL BatchResult SGD::trainNet(NeuralNet *net, TrainingContext *context,
        float const*input, float const*expectedOutput) {
    ExpectedData expectedData(net, expectedOutput);
    return this->trainNet(net, context, input, &expectedData);
}
VIRTUAL BatchResult SGD::trainNetFromLabels(NeuralNet *net, TrainingContext *context,
        float const*input, int const*labels) {
    LabeledData labeledData(net, labels);
    return this->trainNet(net, context, input, &labeledData);
}
VIRTUAL void SGD::bindState(NeuralNet *net) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: bindState");
#endif


    SGDStateMaker stateMaker;
    this->_bindState(net, &stateMaker);
}
STATIC SGD *SGD::instance(EasyCL *cl, float learningRate) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: instance");
#endif


    SGD *sgd = new SGD(cl);
    sgd->setLearningRate(learningRate);
    return sgd;
}
STATIC SGD *SGD::instance(EasyCL *cl, float learningRate, float momentum) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/SGD.cpp: instance");
#endif


    SGD *sgd = new SGD(cl);
    sgd->setLearningRate(learningRate);
    sgd->setMomentum(momentum);
    return sgd;
}
SGD::SGD(EasyCL *cl) :
        Trainer(cl),
        momentum(0.0f),
        weightDecay(0.0f),setup(false) {
}

