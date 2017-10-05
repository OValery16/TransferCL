// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../util/stringhelper.h"
#include "PoolingForwardCpu.h"
#include "PoolingForwardGpuNaive.h"

#include "PoolingForward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingForward::PoolingForward(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) :
        cl(cl),
        padZeros(padZeros),
        numPlanes(numPlanes),
        inputSize(inputSize),
        poolingSize(poolingSize),
        outputSize(padZeros ? (inputSize + poolingSize - 1) / poolingSize : inputSize / poolingSize) {
//    if(inputSize % poolingSize != 0) {
//        throw runtime_error("inputSize should be an exact multiple of poolingsize: " + toString(inputSize) + " " + toString(poolingSize) );
//    }
}
STATIC PoolingForward *PoolingForward::instance(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: instance");
#endif


    return new PoolingForwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize);
//    return new PoolingForwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
}
STATIC PoolingForward *PoolingForward::instanceForTest(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: instanceForTest");
#endif


    return new PoolingForwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize);
}
STATIC PoolingForward *PoolingForward::instanceSpecific(int idx, EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: instanceSpecific");
#endif


    if(idx == 0) {
        return new PoolingForwardCpu(cl, padZeros, numPlanes, inputSize, poolingSize);
    }
    if(idx == 1) {
        return new PoolingForwardGpuNaive(cl, padZeros, numPlanes, inputSize, poolingSize);
    }
    cout << "idx " << idx << " not known" << endl;
    throw runtime_error("PoolingForward::instanceSpecific idx not known: " + toString(idx) );
}
VIRTUAL void PoolingForward::forward(int batchSize, CLWrapper *inputData, CLWrapper *selectors, CLWrapper *outputData) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: forward");
#endif


    throw runtime_error("forward not implemented for this child type");
}
VIRTUAL void PoolingForward::forward(int batchSize, float *input, int *selectors, float *output) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: forward");
#endif


//    cout << "PoolingForward::forward(float *)" << endl;
    CLWrapper *inputWrapper = cl->wrap(getInputNumElements(batchSize), input);
    CLWrapper *selectorsWrapper = cl->wrap(getOutputNumElements(batchSize), selectors);
    CLWrapper *outputWrapper = cl->wrap(getOutputNumElements(batchSize), output);

    inputWrapper->copyToDevice();
    forward(batchSize, inputWrapper, selectorsWrapper, outputWrapper);
    selectorsWrapper->copyToHost();    
    outputWrapper->copyToHost();    

    delete outputWrapper;
    delete selectorsWrapper;
    delete inputWrapper;
}
VIRTUAL int PoolingForward::getInputNumElements(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: getInputNumElements");
#endif


    return batchSize * numPlanes * inputSize * inputSize;
}
VIRTUAL int PoolingForward::getOutputNumElements(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingForward.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}


