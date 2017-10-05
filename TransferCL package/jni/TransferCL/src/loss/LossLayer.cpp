// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "LossLayer.h"
#include "IAcceptsLabels.h"
#include "../batch/BatchData.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

LossLayer::LossLayer(Layer *previousLayer, LossLayerMaker *maker) :
        Layer(previousLayer, maker) {
}
VIRTUAL void LossLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: forward");
#endif


}
VIRTUAL bool LossLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: needsBackProp");
#endif


    return previousLayer->needsBackProp();
}
VIRTUAL float *LossLayer::getOutput() {
#if 1//TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getOutput");
#endif


    return previousLayer->getOutput();
}
VIRTUAL int LossLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getOutputNumElements");
#endif


    return previousLayer->getOutputNumElements();
}
VIRTUAL int LossLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getOutputCubeSize");
#endif


    return previousLayer->getOutputCubeSize();
}
VIRTUAL int LossLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getOutputSize");
#endif


    return previousLayer->getOutputSize();
}
VIRTUAL int LossLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getOutputPlanes");
#endif


    return previousLayer->getOutputPlanes();
}
VIRTUAL int LossLayer::getWeightsSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: getWeightsSize");
#endif


    return previousLayer->getWeightsSize();
}

VIRTUAL float LossLayer::calcLoss(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: calcLoss");
#endif

    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        return this->calcLoss(expectedData->expected);
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        return labeled->calcLossFromLabels(labeledData->labels);
    } else {
    	LOGE( "OutputData child class not implemeneted in LossLayer::calcLoss");
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcLoss");
    }
}

VIRTUAL void LossLayer::calcGradInput(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: calcGradInput");
#endif


    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        this->calcGradInput(expectedData->expected);
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        labeled->calcGradInputFromLabels(labeledData->labels);
    } else {
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcGradInput");
    }
}

VIRTUAL int LossLayer::calcNumRight(OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/LossLayer.cpp: calcNumRight");
#endif


    ExpectedData *expectedData = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeledData = dynamic_cast< LabeledData * >(outputData);
    if(expectedData != 0) {
        return 0; // how are we going to calculate num right, if not labeled?
    } else if(labeledData != 0) {
        IAcceptsLabels *labeled = dynamic_cast< IAcceptsLabels * >(this);
        return labeled->calcNumRightFromLabels(labeledData->labels);
    } else {
        throw runtime_error("OutputData child class not implemeneted in LossLayer::calcNumRight");
    }
}

