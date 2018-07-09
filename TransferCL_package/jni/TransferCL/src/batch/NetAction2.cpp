// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "../net/Trainable.h"
#include "NetAction2.h"
#include "Batcher2.h"
#include "../trainers/Trainer.h"
#include "../trainers/TrainingContext.h"
#include "NetAction.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

void NetLearnAction2::run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetAction2.cpp: run1");
#endif


//    cout << "NetLearnLabeledBatch learningrate=" << learningRate << endl;
    TrainingContext context(epoch, batch);
    ExpectedData *expected = dynamic_cast< ExpectedData * >(outputData);
    LabeledData *labeled = dynamic_cast< LabeledData * >(outputData);
    BatchResult batchResult;
    if(expected != 0) {
        batchResult = trainer->train(net, &context, inputData->inputs, expected->expected);
    } else if(labeled != 0) {
        batchResult = trainer->trainFromLabels(net, &context, inputData->inputs, labeled->labels);        
    }
    epochLoss += batchResult.loss;
    epochNumRight += batchResult.numRight;
}

void NetForwardAction2::run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/NetAction2.cpp: run2");
#endif


//    cout << "NetForwardBatch" << endl;
    net->forward(inputData->inputs);
//    trainer->train(net, batchData, batchLabels);
}

//void NetBackpropAction::run(Trainable *net, InputData *inputData, OutputData *outputData) {
////    cout << "NetBackpropBatch learningrate=" << learningRate << endl;
//    net->backwardFromLabels(learningRate, batchLabels);
//}


