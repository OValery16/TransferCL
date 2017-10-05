// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "CrossEntropyLoss.h"
#include "LossLayer.h"

using namespace std;

#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

CrossEntropyLoss::CrossEntropyLoss(Layer *previousLayer, CrossEntropyLossMaker *maker) :
        LossLayer(previousLayer, maker),
        gradInput(0),
        allocatedSize(0) {
}
VIRTUAL CrossEntropyLoss::~CrossEntropyLoss(){
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: ~CrossEntropyLoss");
#endif


    if(gradInput != 0) {
        delete[] gradInput;
    }
}
VIRTUAL std::string CrossEntropyLoss::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: string CrossEntropyLoss::getClassName");
#endif


    return "CrossEntropyLoss";
}
VIRTUAL float*CrossEntropyLoss::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL int CrossEntropyLoss::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL float CrossEntropyLoss::calcLoss(float const *expected) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: calcLoss");
#endif


    float loss = 0;
    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
//    cout << "CrossEntropyLoss::calcLoss" << endl;
    for(int i = 0; i < inputNumElements; i++) {
        float expectedOutput = expected[i];
        float inputValue = input[i];
        float negthisloss = expectedOutput * log(inputValue) 
            + (1 - expectedOutput) * log(1 - inputValue);
        loss -= negthisloss;
    }
    return loss;
 }
VIRTUAL void CrossEntropyLoss::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    gradInput = new float[ batchSize * previousLayer->getOutputNumElements() ];
    this->batchSize = batchSize;
    allocatedSize = batchSize;
}
// just do naively for now, then add sigmoid short-cutting later
VIRTUAL void CrossEntropyLoss::calcGradInput(float const*expectedOutput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loss/CrossEntropyLoss.cpp: calcGradInput");
#endif


    int inputNumElements = previousLayer->getOutputNumElements();
    float *input = previousLayer->getOutput();
    for(int i = 0; i < inputNumElements; i++) {
        gradInput[i] = (input[i] - expectedOutput[i]) / input[i] / (1.0f - input[i]);
    }
}

