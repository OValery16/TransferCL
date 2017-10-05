// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "../net/NeuralNet.h"
#include "FullyConnectedMaker.h"
#include "FullyConnectedLayer.h"
#include "../conv/ConvolutionalLayer.h"
#include "../conv/ConvolutionalMaker.h"

#include "../dependencies.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

FullyConnectedLayer::FullyConnectedLayer(EasyCL *cl, Layer *previousLayer, FullyConnectedMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes(maker->_numPlanes),
        imageSize(maker->_imageSize),
//        fn(maker->_activationFunction),
        batchSize(0) {
    ConvolutionalMaker *convolutionalMaker = new ConvolutionalMaker();
    convolutionalMaker->numFilters(numPlanes * imageSize * imageSize)
                      ->filterSize(previousLayer->getOutputSize())
                        ->biased(maker->_biased)
                        ->weightsInitializer(maker->_weightsInitializer)
                        ->activationLayer(maker->_activationLayer);
    convolutionalMaker->_stride=1;
    convolutionalMaker->_isLast=maker->_isLast;
    convolutionalMaker->batchSize(maker->_batchSize);
    convolutionalMaker->useMaxPooling(false);
	convolutionalMaker->maxPool_spatialExtent(1);
	convolutionalMaker->maxPool_strides(1);
    convolutionalLayer = new ConvolutionalLayer(cl, previousLayer, convolutionalMaker,"fc");
//    delete convolutionalMaker;
}

VIRTUAL FullyConnectedLayer::~FullyConnectedLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: ~FullyConnectedLayer");
#endif


    delete convolutionalLayer;
}
VIRTUAL std::string FullyConnectedLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: string FullyConnectedLayer::getClassName");
#endif


    return "FullyConnectedLayer";
}
VIRTUAL void FullyConnectedLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: setBatchSize");
#endif


#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\layer\\FullyConnectedLayer.cpp: FullyConnectedLayer setBatchSize");
#endif

    convolutionalLayer->previousLayer = this->previousLayer;
    convolutionalLayer->nextLayer = this->nextLayer;
    convolutionalLayer->setBatchSize(batchSize);
    this->batchSize = batchSize;
}
VIRTUAL int FullyConnectedLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutputCubeSize");
#endif


    return numPlanes * imageSize * imageSize;
}
VIRTUAL int FullyConnectedLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutputSize");
#endif


    return imageSize;
}
VIRTUAL int FullyConnectedLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutputPlanes");
#endif


    return numPlanes;
}
VIRTUAL int FullyConnectedLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getPersistSize");
#endif


    return convolutionalLayer->getPersistSize(version);
}
VIRTUAL void FullyConnectedLayer::persistToArray(int version, float *array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: persistToArray");
#endif


    convolutionalLayer->persistToArray(version, array);
}
VIRTUAL void FullyConnectedLayer::unpersistFromArray(int version, float const*array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: unpersistFromArray");
#endif


    convolutionalLayer->unpersistFromArray(version, array);
}
VIRTUAL void FullyConnectedLayer::setWeights(float *weights, float *bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: setWeights");
#endif


    convolutionalLayer->initWeights(weights);
    convolutionalLayer->initBias(bias);
}
VIRTUAL float * FullyConnectedLayer::getWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getWeights");
#endif


    return convolutionalLayer->getWeights();
}
VIRTUAL float * FullyConnectedLayer::getBias() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getBias");
#endif


    return convolutionalLayer->getBias();
}
VIRTUAL int FullyConnectedLayer::getWeightsSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getWeightsSize");
#endif


    return convolutionalLayer->getWeightsSize();
}
VIRTUAL int FullyConnectedLayer::getBiasSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getBiasSize");
#endif


    return convolutionalLayer->getBiasSize();
}
VIRTUAL int FullyConnectedLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutputNumElements");
#endif


    return convolutionalLayer->getOutputNumElements();
}
VIRTUAL float *FullyConnectedLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutput");
#endif


    return convolutionalLayer->getOutput();
}
VIRTUAL float *FullyConnectedLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getGradInput");
#endif


    return convolutionalLayer->getGradInput();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getGradWeightsWrapper");
#endif


    return convolutionalLayer->getGradWeightsWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getGradBiasWrapper");
#endif


    return convolutionalLayer->getGradBiasWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getWeightsWrapper");
#endif


    return convolutionalLayer->getWeightsWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getBiasWrapper");
#endif


    return convolutionalLayer->getBiasWrapper();
}
VIRTUAL bool FullyConnectedLayer::biased() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: biased");
#endif


    return convolutionalLayer->biased();
}
VIRTUAL bool FullyConnectedLayer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: providesGradInputWrapper");
#endif


    return convolutionalLayer->providesGradInputWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getGradInputWrapper");
#endif


    return convolutionalLayer->getGradInputWrapper();
}
VIRTUAL bool FullyConnectedLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: hasOutputWrapper");
#endif


    return convolutionalLayer->hasOutputWrapper();
}
VIRTUAL CLWrapper *FullyConnectedLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getOutputWrapper");
#endif


    return convolutionalLayer->getOutputWrapper();
}
//VIRTUAL ActivationFunction const*FullyConnectedLayer::getActivationFunction() {
//    return fn;
//}
VIRTUAL bool FullyConnectedLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: needsBackProp");
#endif


    return true;;
}
VIRTUAL void FullyConnectedLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: forward");
#endif


    convolutionalLayer->forward();
}
VIRTUAL void FullyConnectedLayer::backward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: backward");
#endif


    convolutionalLayer->backward();
}
VIRTUAL bool FullyConnectedLayer::needsTrainerState() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: needsTrainerState");
#endif


    return true;
}
VIRTUAL TrainerState *FullyConnectedLayer::getTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getTrainerState");
#endif


    return convolutionalLayer->getTrainerState();
}
VIRTUAL TrainerState *FullyConnectedLayer::getBiasTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: getBiasTrainerState");
#endif


    return convolutionalLayer->getBiasTrainerState();
}
VIRTUAL void FullyConnectedLayer::setTrainerState(TrainerStateMaker *TrainerStateMaker) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: setTrainerState");
#endif


    convolutionalLayer->setTrainerState(TrainerStateMaker);
}
VIRTUAL std::string FullyConnectedLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/fc/FullyConnectedLayer.cpp: string FullyConnectedLayer::asString");
#endif


    return "FullyConnectedLayer{ numPlanes=" + toString(numPlanes) + " imageSize=" + toString(imageSize) + " }";
}

