// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "SquareLossLayer.h"
#include "LossLayer.h"
#include "../layer/LayerMaker.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

SquareLossLayer::SquareLossLayer(Layer *previousLayer, SquareLossMaker *maker) :
        LossLayer(previousLayer, maker),
        gradInput(0),
        allocatedSize(0) {
}
VIRTUAL SquareLossLayer::~SquareLossLayer(){
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: ~SquareLossLayer");
#endif


    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string SquareLossLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: string SquareLossLayer::getClassName");
#endif


    return "SquareLossLayer";
}
VIRTUAL float*SquareLossLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL float SquareLossLayer::calcLoss(float const *expected) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: calcLoss");
#endif


    float loss = 0;
//    float *output = getOutput();
    float *input = previousLayer->getOutput();
//    cout << "SquareLossLayer::calcLoss" << endl;
    int numPlanes = previousLayer->getOutputPlanes();
    int imageSize = previousLayer->getOutputSize();
    int totalLinearSize = batchSize * numPlanes * imageSize * imageSize;
    for(int i = 0; i < totalLinearSize; i++) {
//        if(i < 5) cout << "input[" << i << "]=" << input[i] << endl;
        float diff = input[i] - expected[i];
//        LOGI( "diff=%f",input[i]);
        float diffSquared = diff * diff;
        loss += diffSquared;
    }
    loss *= 0.5f;
//    cout << "loss " << loss << endl;
    return loss;
 }
VIRTUAL void SquareLossLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    this->batchSize = batchSize;
    allocatedSize = batchSize;
    gradInput = new float[ batchSize * previousLayer->getOutputNumElements() ];
}
VIRTUAL void SquareLossLayer::calcGradInput(float const*expectedOutput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: calcGradInput");
#endif


    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < inputNumElements; i++) {
        gradInput[i] = input[i] - expectedOutput[i];
    }
}
VIRTUAL int SquareLossLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL std::string SquareLossLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/SquareLossLayer.cpp: string SquareLossLayer::asString");
#endif


    return "SquareLossLayer{}";
}

