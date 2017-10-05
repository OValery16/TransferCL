// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>

#include "../../EasyCL/EasyCL.h"
#include "../util/stringhelper.h"
#include "../../EasyCL/util/StatefulTimer.h"

#include "PoolingBackwardCpu.h"
#include "PoolingBackwardGpuNaive.h"

#include "PoolingBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

STATIC PoolingBackward *PoolingBackward::instance(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize,int previousLayer_activationLayer,bool test) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: instance");
#endif

    return new PoolingBackwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize,previousLayer_activationLayer,test);
}
STATIC PoolingBackward *PoolingBackward::instanceForTest(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: instanceForTest");
#endif


    return new PoolingBackwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
}
STATIC PoolingBackward *PoolingBackward::instanceSpecific(int idx, EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: instanceSpecific");
#endif

int temp=0;


    if(idx == 0) {
        return new PoolingBackwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
    }
    if(idx == 1) {
        return new PoolingBackwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize,temp,0);
    }
    throw runtime_error("PoolingBackward::instanceSpecific, idx not known: " + toString(idx) );
}
PoolingBackward::PoolingBackward(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) :
        cl(cl),
        padZeros(padZeros),
        numPlanes(numPlanes),
        inputSize(inputSize),
        poolingSize(poolingSize),
//        poolingSizeSquared(poolingSize * poolingSize),
        outputSize(padZeros ? (inputSize + poolingSize - 1) / poolingSize : inputSize / poolingSize) {
//    if(inputSize % poolingSize != 0) {
//        throw runtime_error("inputSize should be an exact multiple of poolingsize: " + toString(inputSize) + " " + toString(poolingSize) );
//    }
}
VIRTUAL int PoolingBackward::getInputNumElements(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: getInputNumElements");
#endif


    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int PoolingBackward::getOutputNumElements(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL void PoolingBackward::backward(int batchSize, float *gradOutput, int *selectors, float *gradInput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: backward");
#endif


//    cout << "PoolingBackward::backward(float *)" << endl;
    StatefulTimer::instance()->timeCheck("PoolingBackward::backward float->wrapper start");
    CLWrapper *gradOutputWrapper = cl->wrap(getOutputNumElements(batchSize), gradOutput);
    CLWrapper *selectorsWrapper = cl->wrap(getOutputNumElements(batchSize), selectors);
    CLWrapper *gradInputWrapper = cl->wrap(getInputNumElements(batchSize), gradInput);

    gradOutputWrapper->copyToDevice();
    selectorsWrapper->copyToDevice();
    CLWrapper *inputWrapper=0;//olivier we don t use that path

    backward(batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper, inputWrapper);

    selectorsWrapper->copyToHost();
    gradInputWrapper->copyToHost();

    delete gradOutputWrapper;
    delete selectorsWrapper;
    delete gradInputWrapper;
    StatefulTimer::instance()->timeCheck("PoolingBackward::backward float->wrapper end");
}
VIRTUAL void PoolingBackward::backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper, CLWrapper *gradInputWrapper,CLWrapper * inputWrapper) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackward.cpp: backward");
#endif


    throw runtime_error("PoolingBackward::backward wrappers not implemented");
}

