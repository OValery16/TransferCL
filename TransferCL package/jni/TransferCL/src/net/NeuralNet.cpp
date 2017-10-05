// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>
#include <random>

#include "../util/Timer.h"
#include "../conv/ConvolutionalLayer.h"
#include "../layer/LayerMaker.h"
#include "NeuralNetMould.h"
#include "../activate/ActivationFunction.h"
#include "../../EasyCL/util/StatefulTimer.h"
//#include "AccuracyHelper.h"
#include "../layer/Layer.h"
#include "../input/InputLayer.h"
#include "../fc/FullyConnectedLayer.h"
#include "../batch/EpochMaker.h"
#include "../loss/LossLayer.h"
#include "../loss/IAcceptsLabels.h"
#include "../util/ExceptionMacros.h"
#include "../input/InputLayerMaker.h"
#include "../trainers/Trainer.h"
#include "../trainers/TrainerMaker.h"
#include "../weights/WeightsPersister.h"
#include "../CppRuntimeBoundary.h"

#include "../dependencies.h"

#include "NeuralNet.h"

using namespace std;

#define MEASURE_BACKWARD_PROP 0
#define MEASURE_FORWARD_PROP 0

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

NeuralNet::NeuralNet(EasyCL *cl) :
        cl(cl) {
    trainer = 0;
    isTraining = true;
}
STATIC NeuralNet *NeuralNet::instance(EasyCL *cl) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: instance");
#endif


    return new NeuralNet(cl);
}
STATIC NeuralNet *NeuralNet::instance(EasyCL *cl, int numPlanes, int imageSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: instance");
#endif


    return new NeuralNet(cl, numPlanes, imageSize);
}
STATIC NeuralNet *NeuralNet::instance3(EasyCL *cl, int numPlanes, int imageSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: instance3");
#endif


    return new NeuralNet(cl, numPlanes, imageSize);
}
void NeuralNet::deleteMe() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: deleteMe");
#endif


    delete this;
}
/// Constructor
NeuralNet::NeuralNet(EasyCL *cl, int numPlanes, int imageSize) :
        cl(cl) {
    addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize) );
    trainer = 0;
}
NeuralNet::~NeuralNet() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: ~NeuralNet");
#endif

#pragma omp parallel for
    for(int i = 0; i < (int)layers.size(); i++) {
    	//LOGI("%d",i);
        delete layers[i];

    }
}
STATIC NeuralNetMould *NeuralNet::maker(EasyCL *cl) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: maker");
#endif


    return new NeuralNetMould(cl);
}
NeuralNet *NeuralNet::clone() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: clone");
#endif


    NeuralNet *copy = new NeuralNet(cl);
    for(vector<Layer *>::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        LayerMaker2 *maker = (*it)->maker;

        LayerMaker2 *makerCopy = maker->clone();
        copy->addLayer(makerCopy);
    }
    copy->print();
    cout << "outputimagesize: " << copy->getOutputSize() << endl;
    return copy;
}
EasyCL *NeuralNet::getCl() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getCl");
#endif


    return cl;
}
/// Add a network layer, using a LayerMaker2 object
PUBLICAPI void NeuralNet::addLayer(LayerMaker2 *maker) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: addLayer");
#endif


//    cout << "neuralnet::insert numplanes " << inputLayerMaker._numPlanes << " imageSize " << inputLayerMaker._imageSize << endl;
    maker->setCl(cl);
    Layer *layer = maker->createLayer(getLastLayer());
    layers.push_back(layer);
}
PUBLICAPI void NeuralNet::initWeights(int layerIndex, float *weights, float *bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: initWeights");
#endif


    initWeights(layerIndex, weights);
    initBias(layerIndex, bias);
}
PUBLICAPI void NeuralNet::initWeights(int layerIndex, float *weights) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: initWeights");
#endif


    layers[layerIndex]->initWeights(weights);
}
PUBLICAPI void NeuralNet::initBias(int layerIndex, float *weights) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: initBias");
#endif


    layers[layerIndex]->initBias(weights);
}
/// \brief calculate the loss, based on the passed in expectedValues array
///
/// \publicapi
///
/// Calculate the loss, based on the passed in expectedValues array
/// which should be the same size as the output of the final layer
/// of the network
PUBLICAPI float NeuralNet::calcLoss(float const *expectedValues) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: calcLoss");
#endif


    return dynamic_cast<LossLayer*>(getLastLayer())->calcLoss(expectedValues);
}
PUBLICAPI float NeuralNet::calcLossFromLabels(int const *labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: calcLossFromLabels");
#endif


    return dynamic_cast<IAcceptsLabels*>(getLastLayer())->calcLossFromLabels(labels);
}
float NeuralNet::calcLoss(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: calcLoss");
#endif


    return dynamic_cast<LossLayer*>(getLastLayer())->calcLoss(outputData);
}
int NeuralNet::calcNumRight(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: calcNumRight");
#endif


    return dynamic_cast<LossLayer*>(getLastLayer())->calcNumRight(outputData);
}
EpochMaker *NeuralNet::epochMaker(Trainer *trainer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: epochMaker");
#endif


     return new EpochMaker(this, trainer);
}
VIRTUAL LossLayerMaker *NeuralNet::cloneLossLayerMaker() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: cloneLossLayerMaker");
#endif


    LossLayer const *lossLayer = dynamic_cast< LossLayer const*>(getLastLayer());
    if(lossLayer == 0) {
        throw runtime_error("error: last layer must be a losslayer");
    }
    return dynamic_cast< LossLayerMaker *>(lossLayer->maker->clone());
//    throw runtime_error("need to implement neuralnet::clonelosslayermaker :-)");
//    LossLayer const*lossLayer = dynamic_cast< LossLayer const*>(getLastLayer());
//    return dynamic_cast< LossLayerMaker *>(lossLayer->maker->clone(clonePreviousLayer) ) ;
}
PUBLICAPI InputLayer *NeuralNet::getFirstLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getFirstLayer");
#endif


    return dynamic_cast<InputLayer *>(layers[0]);
}
PUBLICAPI Layer *NeuralNet::getLastLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getLastLayer");
#endif


    if(layers.size() == 0) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI int NeuralNet::getNumLayers() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getNumLayers");
#endif


    return (int)layers.size();
}
PUBLICAPI Layer *NeuralNet::getLayer(int index) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getLayer");
#endif


    if(layers.size() == 0) {
        return 0;
    }
    if(index < 0 || index > (int)layers.size() - 1) {
        return 0;
    }
    return layers[index];
}
PUBLICAPI Layer const*NeuralNet::getLastLayer() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getLastLayer");
#endif


    if(layers.size() == 0) {
        return 0;
    }
    return layers[layers.size() - 1];
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutputPlanes");
#endif


    return getLastLayer()->getOutputPlanes();
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutputSize");
#endif


    return getLastLayer()->getOutputSize();
}
PUBLICAPI void NeuralNet::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: setBatchSize");
#endif


#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\net\\NeuralNet.cpp: neural net setBatchSize");
#endif
int i=0;
    for(std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
		LOGI("%d",i);
#endif

		i++;
        (*it)->setBatchSize(batchSize);
    }
}
PUBLICAPI void NeuralNet::setTraining(bool training) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: setTraining");
#endif


#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\net\\NeuralNet.cpp: neural net setTraining");
#endif

    for(std::vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector<Layer*>::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        (*it)->setTraining(training);
    }
}
PUBLICAPI int NeuralNet::calcNumRight(int const *labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: calcNumRight");
#endif


    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if(acceptsLabels == 0) {
        THROW("You need to add a IAcceptsLabels as the last layer, in order to use calcNumRight");
    }
    return acceptsLabels->calcNumRightFromLabels(labels);
}
PUBLICAPI void NeuralNet::forward(float const*images) {
	#if TRANSFERCL_VERBOSE == 1
	LOGI( "--------------DeepCL/src/net/NeuralNet.cpp: forward");
	#endif

	#if MEASURE_FORWARD_PROP==1
		struct timeval start1, end1;


//	LOGI( "network size %d",(int)layers.size());
//	clock_t startTimer1, stopTimer1;
	#endif

    // forward...
    dynamic_cast<InputLayer *>(layers[0])->in(images);
    for(int layerId = 0; layerId < (int)layers.size(); layerId++) {
		#if MEASURE_FORWARD_PROP==1
//			startTimer1=clock();
//			StatefulTimer::setPrefix("layer" + toString(layerId) + " ");
    		gettimeofday(&start1, NULL);
		#endif
        layers[layerId]->forward();
		#if MEASURE_FORWARD_PROP==1
            cl->finish();
            gettimeofday(&end1, NULL);
            LOGI("-----------------forward (layerId %d) took %f\n ms",layerId, (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));

//			stopTimer1=clock();
//			LOGI("layer %d took %g ms\n\n",layerId,1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
//			StatefulTimer::setPrefix("");
		#endif
    }
}
/// \brief note: this does no learning, just calculates the gradients
PUBLICAPI void NeuralNet::backwardFromLabels(int const *labels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: backwardFromLabels");
#endif


    IAcceptsLabels *acceptsLabels = dynamic_cast<IAcceptsLabels*>(getLastLayer());
    if(acceptsLabels == 0) {
        throw std::runtime_error("Must add a child of IAcceptsLabels as last layer, to use backwardFromLabels");
    }
    acceptsLabels->calcGradInputFromLabels(labels);
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer :-P
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        Layer *layer = layers[layerIdx];
        if(layer->needsBackProp()) {
        	LOGI("needsBackProp");
            layer->backward();
        }
        StatefulTimer::setPrefix("");
    }
}
/// \brief note: this does no learning, just calculates the gradients
PUBLICAPI void NeuralNet::backward(float const *expectedOutput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: backward");
#endif


    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    if(lossLayer == 0) {
        throw std::runtime_error("Must add a LossLayer as last layer of net");
    }
    lossLayer->calcGradInput(expectedOutput);
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer
    	LOGI("backward id = %d", layerIdx);
        StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        layers[layerIdx]->backward();
        StatefulTimer::setPrefix("");
    }
}
void NeuralNet::backward(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGE( "DeepCL/src/net/NeuralNet.cpp: backward");
#endif

	#if MEASURE_BACKWARD_PROP==1
		struct timeval start1, end1;
		gettimeofday(&start1, NULL);
//		clock_t startTimer1, stopTimer1;
//		startTimer1=clock();
	#endif


    LossLayer *lossLayer = dynamic_cast<LossLayer*>(getLastLayer());
    lossLayer->calcGradInput(outputData);
	#if MEASURE_BACKWARD_PROP==1
    	gettimeofday(&end1, NULL);
        LOGI("-----------------calcGrad took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));

//		stopTimer1=clock();
//		LOGI("calcGradInput took %g ms\n\n",1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
	#endif

#if TRANSFER ==0
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= 1; layerIdx--) { // no point in propagating to input layer
#endif
#if TRANSFER ==1
    for(int layerIdx = (int)layers.size() - 2; layerIdx >= (int)layers.size()-3; layerIdx--) { // no point in propagating to input layer
#endif

		#if MEASURE_BACKWARD_PROP==1
    	    cl->finish();
    	    gettimeofday(&start1, NULL);
//			startTimer1=clock();
//			LOGI("backward2 id = %d", layerIdx);
		#endif
        Layer *layer = getLayer(layerIdx);
        if(!layer->needsBackProp()) {
			#if MEASURE_BACKWARD_PROP==1
				LOGI("exit");
			#endif
            break;
        }
        //StatefulTimer::setPrefix("layer" + toString(layerIdx) + " ");
        layer->backward();
        //StatefulTimer::setPrefix("");
		#if MEASURE_BACKWARD_PROP==1
        	cl->finish();
        	gettimeofday(&end1, NULL);
            LOGI("-----------------backward (layerId %d) took %f\n ms",layerIdx, (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));

//			stopTimer1=clock();
//			LOGI("--------------------------------------------------layer %d took %g ms\n\n",layerIdx,1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		#endif
    }
}
PUBLICAPI int NeuralNet::getNumLayers() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getNumLayers");
#endif


    return (int)layers.size();
}
PUBLICAPI float const *NeuralNet::getOutput(int layer) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutput");
#endif


    return layers[layer]->getOutput();
}
PUBLICAPI int NeuralNet::getInputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getInputCubeSize");
#endif


    return layers[ 0 ]->getOutputCubeSize();
}
PUBLICAPI int NeuralNet::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutputCubeSize");
#endif


    return layers[ layers.size() - 1 ]->getOutputCubeSize();
}
PUBLICAPI float const *NeuralNet::getOutput() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutput");
#endif


    return getOutput((int)layers.size() - 1);
}
PUBLICAPI VIRTUAL int NeuralNet::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: getOutputNumElements");
#endif


    return getLastLayer()->getOutputNumElements();
}
void NeuralNet::print() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: print");
#endif


    cout << this->asString();
    printParamStats();
//    int i = 0; 
//    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
//        std::cout << "layer " << i << ":" << (*it)->asString() << endl;
//        i++;
//    }
}
void NeuralNet::printWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: printWeights");
#endif


    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printWeights();
        i++;
    }
}
void NeuralNet::printOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: printOutput");
#endif


    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        std::cout << "layer " << i << ":" << std::endl;
        (*it)->printOutput();
        i++;
    }
}
VIRTUAL void NeuralNet::setTrainer(Trainer *trainer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: setTrainer");
#endif


    this->trainer = trainer;
}
void NeuralNet::printParamStats() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: printParamStats");
#endif


    int sum = 0;
    int skip = 0;
    int precision = (int)std::cout.precision();
//    cout << "precision: " << precision << endl;
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        int size = (*it)->getPersistSize(WeightsPersister::latestVersion);
        sum += size;
        if(! size){
            skip++;
        }
    }
    std::cout << "Parameters overview: (skipping " << skip << " layers with 0 params)" << std::endl;
    int i = 0;
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++, i++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++, i++) {");
#endif


        int size = (*it)->getPersistSize(WeightsPersister::latestVersion);
        if(size) {
            std::cout << "layer " << i << ": params=" << size << "\t";
            std::cout << std::fixed << std::setprecision(1) << ((float) 100 * size)/sum << "%";
            std::cout << std::endl;
        }
    }
    if(i){
        std::cout << "TOTAL  : params=" << sum << std::endl;
    }
    // reset the cout properties, so that I dont spend 2 hours figuring out why my weights
    // all changed to 0.0 and 0.1 :-P
    std::cout << setprecision(precision);
    std::cout.unsetf(ios_base::floatfield);
}
PUBLICAPI std::string NeuralNet::asString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: string NeuralNet::asString");
#endif


    std::string result = "";
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        result += "layer " + toString(i) + ":" + (*it)->asString() + "\n";
        i++;
    }    
    return result;
}
PUBLICAPI const char * NeuralNet::asNewCharStar() { // call deepcl_deleteCharStar to delete this
    std::string result = "";
    int i = 0; 
    for(std::vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNet.cpp: vector< Layer* >::iterator it = layers.begin(); it != layers.end(); it++) {");
#endif


        result += "layer " + toString(i) + ":" + (*it)->asString() + "\n";
        i++;
    }
    return deepcl_stringToCharStar(result);
}

