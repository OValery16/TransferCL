// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "Layer.h"
#include "../weights/WeightsPersister.h"
#include "../CppRuntimeBoundary.h"

#include "../dependencies.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

PUBLICAPI Layer::Layer(Layer *previousLayer, LayerMaker2 *maker) :
    previousLayer(previousLayer),
    nextLayer(0),
    layerIndex(previousLayer == 0 ? 0 : previousLayer->layerIndex + 1),
    training(false),
    maker(maker),
    momentum(0.0f),
    learning_rate(0.0f)
     {
	//LOGI("create layer PUBLICAPI Layer::Layer(Layer *previousLayer, LayerMaker2 *maker) :");
    if(previousLayer != 0) {
        previousLayer->nextLayer = this;
    }
}

VIRTUAL Layer::~Layer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: ~Layer");
#endif


    if(maker != 0) {
        //delete maker; // this segfaults sometimes, (probably because it already
                        // self-deleted)
    }
}

PUBLICAPI VIRTUAL void Layer::setMomentum(float _momentum) {
	this->momentum=_momentum;
}

PUBLICAPI VIRTUAL void Layer::setLearningRate(float _learning_rate) {
	this->learning_rate=_learning_rate;
}
PUBLICAPI VIRTUAL void Layer::setWeightDecay(float _weightDecay) {
	this->weightDecay=_weightDecay;
}

/// \brief Are we training or predicting?
/// Only affects the Random translations and patches layers currently
PUBLICAPI VIRTUAL void Layer::setTraining(bool training) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: setTraining");
#endif


#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\layer\\layer.cpp: layer setTraining");
#endif

    this->training = training;
}
/// used to set up internal buffers and stuff
PUBLICAPI VIRTUAL void Layer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: setBatchSize");
#endif


#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\layer\\layer.cpp: neural net setBatchSize");
#endif

#if CLMATH_VERBOSE == 1
	LOGE("setBatchsize not implemetned for this layer type");
#endif

    throw std::runtime_error("setBatchsize not implemetned for this layer type");
}
VIRTUAL bool Layer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: providesGradInputWrapper");
#endif


    return false;
}
VIRTUAL const char *Layer::getClassNameAsCharStar() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getClassNameAsCharStar");
#endif


    return deepcl_stringToCharStar(getClassName());
}
VIRTUAL float *Layer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradInput");
#endif


    throw std::runtime_error("getGradInput not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradWeightsWrapper");
#endif


    throw std::runtime_error("getGradWeightsWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradBiasWrapper");
#endif


    throw std::runtime_error("getGradBiasWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getWeightsWrapper");
#endif


    throw std::runtime_error("getWeightsWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBiasWrapper");
#endif


    throw std::runtime_error("getBiasWrapper not implemented for " + getClassName());
}
VIRTUAL CLWrapper *Layer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradInputWrapper");
#endif


    throw std::runtime_error("getGradInputWrapper not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL bool Layer::getBiased() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBiased");
#endif


     throw std::runtime_error("getBiased not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL bool Layer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: hasOutputWrapper");
#endif


    return false;
}

PUBLICAPI VIRTUAL bool Layer::isFirstLayer() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: isFirstLayer");
#endif


    return false;
}
PUBLICAPI VIRTUAL bool Layer::isConvLayer() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: isConvLayer");
#endif


    return false;
}
PUBLICAPI VIRTUAL CLWrapper *Layer::getSelectorWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getSelectorWrapper");
#endif

LOGE("getSelectorWrapper not implemetned");
    throw std::runtime_error("getSelectorWrapper not implemetned for " + getClassName());
}

PUBLICAPI VIRTUAL CLWrapper *Layer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputWrapper");
#endif


    throw std::runtime_error("getOutputWrapper not implemetned for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputCubeSize");
#endif


    throw std::runtime_error("getOutputCubeSize not implemetned for " + getClassName());
 //     return numPlanes * imageSize * imageSize * batchSize;
}
PUBLICAPI VIRTUAL int Layer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputPlanes");
#endif


    throw std::runtime_error("getOutputPlanes not implemetned for " + getClassName());
}

PUBLICAPI VIRTUAL float Layer::getTranslate() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputPlanes");
#endif


    return 0.0f;
}

PUBLICAPI VIRTUAL float Layer::getScale() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputPlanes");
#endif


    return 1.0f;
}

PUBLICAPI VIRTUAL int Layer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getOutputSize");
#endif


    throw std::runtime_error("getOutputSize not implemetned for " + getClassName());
}
VIRTUAL void Layer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: forward");
#endif


    throw std::runtime_error("forward not implemented for " + getClassName());
}
VIRTUAL bool Layer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: needsBackProp");
#endif


    throw std::runtime_error("needsBackProp not implemented for " + getClassName());
}
VIRTUAL void Layer::print() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: print");
#endif


//    printWeights();
//    if(output != 0) {
    printOutput();
    printWeights();
//    } else {
//        std::cout << "No output yet " << std::endl;
//    }
}
VIRTUAL void Layer::initWeights(float const*weights) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: initWeights");
#endif


    throw std::runtime_error("initWeights not implemetned for " + getClassName());
//    int numWeights = getWeightsSize();
//    for(int i = 0; i < numWeights; i++) {
//        this->weights[i] = weights[i];
//    }
}
VIRTUAL void Layer::initBias(float const *bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: initBias");
#endif


    throw std::runtime_error("initBias not implemetned for " + getClassName());
//    int numBias = getBiasSize();
//    for(int i = 0; i < numBias; i++) {
//        this->bias[i] = bias[i];
//    }
}
int Layer::getLayerIndex() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getLayerIndex");
#endif


    return layerIndex;
}
VIRTUAL void Layer::printWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: printWeights");
#endif


    throw std::runtime_error("printWeights not implemented for " + getClassName());
}
VIRTUAL void Layer::printOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: printOutput");
#endif


    throw std::runtime_error("printOutput not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL void Layer::backward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: backward");
#endif


    throw std::runtime_error("backward not implemented for " + getClassName());
}
VIRTUAL float *Layer::getGradWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradWeights");
#endif


    throw std::runtime_error("getGradWeights not implemented for " + getClassName());
}


VIRTUAL float *Layer::getGradBias() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getGradBias");
#endif


    throw std::runtime_error("getGradBias not implemented for " + getClassName());
}
VIRTUAL bool Layer::biased() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: biased");
#endif


    throw std::runtime_error("biased not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getWeightsSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getWeightsSize");
#endif


    throw std::runtime_error("getWeightsSize not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getBiasSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBiasSize");
#endif


    throw std::runtime_error("getBiasSize not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL int Layer::getPersistSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getPersistSize");
#endif


    return getPersistSize(WeightsPersister::latestVersion);
}
PUBLICAPI VIRTUAL void Layer::persistToArray(float *array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: persistToArray");
#endif


    persistToArray(WeightsPersister::latestVersion, array);
}
/// \brief store the current weights and biases to array
/// Note that you need to allocate array first
PUBLICAPI VIRTUAL void Layer::persistToArray(int version, float *array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: persistToArray");
#endif


    throw std::runtime_error("persistToArray not implemented for " + getClassName());
}
PUBLICAPI VIRTUAL void Layer::unpersistFromArray(float const*array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: unpersistFromArray");
#endif


    unpersistFromArray(WeightsPersister::latestVersion, array);
}
/// \brief initialize the current weights and biases from array
PUBLICAPI VIRTUAL void Layer::unpersistFromArray(int version, float const*array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: unpersistFromArray");
#endif


    throw std::runtime_error("unpersistFromArray not implemented for " + getClassName());
}
VIRTUAL void Layer::setWeights(float *weights, float *bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: setWeights");
#endif


    throw std::runtime_error("setWeights not implemented for " + getClassName());
}
VIRTUAL float const *Layer::getWeights() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getWeights");
#endif


    throw std::runtime_error("getWeights const not implemented for " + getClassName());
}
VIRTUAL float *Layer::getWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getWeights");
#endif


    throw std::runtime_error("getWeights not implemented for " + getClassName());
}
VIRTUAL float *Layer::getBias() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBias");
#endif


    throw std::runtime_error("getBias not implemented for " + getClassName());
}
VIRTUAL float const*Layer::getBias() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBias");
#endif


    throw std::runtime_error("getBias const not implemented for " + getClassName());
}
/// \brief Get a string representation of the layer
VIRTUAL std::string Layer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: string Layer::asString");
#endif


    return "Layer{}";
}
VIRTUAL const char *Layer::asNewCharStar() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: asNewCharStar");
#endif


    return deepcl_stringToCharStar(asString());
}
VIRTUAL bool Layer::needsTrainerState  () const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: needsTrainerState  ");
#endif


    return false;
}
// This transfers ownership of the trainer to the layer,
// which is responsible for deleting it
// probably should pass in a Maker class instead
VIRTUAL void Layer::setTrainerState(TrainerStateMaker *trainerMaker) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: setTrainerState");
#endif


    throw std::runtime_error("setTrainer not implemented for " + getClassName());
}
VIRTUAL TrainerState *Layer::getTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getTrainerState");
#endif


    throw std::runtime_error("getTrainerState not implemented for " + getClassName());
}
VIRTUAL TrainerState *Layer::getBiasTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: getBiasTrainerState");
#endif


    throw std::runtime_error("getBiasTrainerState not implemented for " + getClassName());
}
VIRTUAL void Layer::updateWeights(CLWrapper *weightChangesWrapper, CLWrapper *biasChangesWrapper) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: updateWeights");
#endif


    throw std::runtime_error("updateWeights not implemented for " + getClassName());
}

