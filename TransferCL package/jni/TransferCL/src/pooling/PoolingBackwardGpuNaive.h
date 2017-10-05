// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include "../conv/ConvolutionalLayer.h"
#include "PoolingBackward.h"
#include "../../EasyCL/templates/TemplatedKernel.h"
#include <cmath>

#define VIRTUAL virtual
#define STATIC static

class PoolingBackwardGpuNaive : public PoolingBackward {
public:
    CLKernel *kernel;
    CLKernel *kMemset;
    CLKernel *kernel2;
    int workgroupSize2;
    int numWorkgroups2 ;
    bool setup;

    bool test;


    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~PoolingBackwardGpuNaive();
    VIRTUAL void backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper,CLWrapper *gradInputWrapper,CLWrapper * inputWrapper);
    PoolingBackwardGpuNaive(EasyCL *cl, bool padZeros, int numPlanes, int inputSize, int poolingSize,int previousLayer_activationLayer,bool test);
    void setupBuilderBackward(TemplatedKernel *builder,int previousLayer_activationLayer);
    void buildKernelBackward( string kernelSource,int previousLayer_activationLayer);
    void  setActivationFunction(TemplatedKernel *builder,int previousLayer_activationLayer);

    // [[[end]]]
};

