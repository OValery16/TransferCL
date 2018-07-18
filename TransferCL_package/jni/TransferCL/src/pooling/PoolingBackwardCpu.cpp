// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "../../EasyCL/EasyCL.h"
#include "PoolingBackward.h"
#include "../../EasyCL/util/StatefulTimer.h"

#include "PoolingBackwardCpu.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PoolingBackwardCpu::PoolingBackwardCpu(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize) :
        PoolingBackward(cl, padZeros, numPlanes, inputSize, poolingSize) {
}
VIRTUAL void PoolingBackwardCpu::backward(int batchSize,  float *gradOutput, int *selectors, float *gradInput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingBackwardCpu.cpp: backward");
#endif


    memset(gradInput, 0, sizeof(float) * getInputNumElements(batchSize) );
    for(int n = 0; n < batchSize; n++) {
        for(int plane = 0; plane < numPlanes; plane++) {
            for(int outputRow = 0; outputRow < outputSize; outputRow++) {
                int inputRow = outputRow * poolingSize;
                for(int outputCol = 0; outputCol < outputSize; outputCol++) {
                    int inputCol = outputCol * poolingSize;
                    int outputIndex = getResultIndex(n, plane, outputRow, outputCol);
                    int selector = selectors[outputIndex];
                    int drow = selector / poolingSize;
                    int dcol = selector % poolingSize;
                    int inputIndex = getInputIndex(n, plane, inputRow + drow, inputCol + dcol);
                    gradInput[ inputIndex ] = gradOutput[outputIndex];
                }
            }
        }
    }
}
VIRTUAL void PoolingBackwardCpu::backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper, 
        CLWrapper *gradInputWrapper) {
    StatefulTimer::instance()->timeCheck("PoolingBackwardCpu::backward start");

    gradOutputWrapper->copyToHost();
    selectorsWrapper->copyToHost();

    float *gradOutput = reinterpret_cast<float *>(gradOutputWrapper->getHostArray());
    int *selectors = reinterpret_cast<int *>(selectorsWrapper->getHostArray());
    float *gradInput = new float[ getInputNumElements(batchSize) ];

    backward(batchSize, gradOutput, selectors, gradInput);

    float *gradInputHostArray = reinterpret_cast<float *>(gradInputWrapper->getHostArray());
    memcpy(gradInputHostArray, gradInput, sizeof(float) * getInputNumElements(batchSize) );
    gradInputWrapper->copyToDevice();

    delete[] gradInput;
    
    StatefulTimer::instance()->timeCheck("PoolingBackwardCpu::backward end");
}

