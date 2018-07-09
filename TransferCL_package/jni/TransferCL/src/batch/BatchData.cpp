// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "BatchData.h"
#include "../net/Trainable.h"

using namespace std;

InputData *InputData::instance(Trainable *net, float const*inputs) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/BatchData.cpp: instance");
#endif


    int inputCubeSize = net->getInputCubeSize();
    return new InputData(inputCubeSize, inputs);
}

ExpectedData *ExpectedData::instance(Trainable *net, float const*expectedOutputs) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/BatchData.cpp: instance");
#endif


    int outputCubeSize = net->getOutputCubeSize();
    return new ExpectedData(outputCubeSize, expectedOutputs);
}
LabeledData *LabeledData::instance(Trainable *net, int const*labels) { // net not used
    // but means dont have to keep remembering whether to add in parameters or not
    return new LabeledData(labels);
}

ExpectedData::ExpectedData(Trainable *net, float const*expected) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/batch/BatchData.cpp: ExpectedData");
#endif


    this->outputCubeSize = net->getOutputCubeSize();
    this->expected = expected;
}
LabeledData::LabeledData(Trainable *net, int const*labels) { // net not used
    // but means dont have to keep remembering whether to add in parameters or not
    this->labels = labels;
}

