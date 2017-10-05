// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include "../dependencies.h"

#include "../../EasyCL/EasyCL.h"
#include "../../EasyCL/templates/TemplatedKernel.h"

#include "../dependencies.h"
#include <iostream>
#include <string>

#include "../../EasyCL/EasyCL.h"
#include "../activate/ActivationFunction.h"
#include "LayerDimensions.h"

#include "../TransferCLDllExport.h"

#define STATIC static
#define VIRTUAL virtual

class BackwardGpuNaive {
public:
    CLKernel *kernel;
    CLKernel *kernel2;

    int globalSize;
    int workgroupsize;

    bool setup;

    EasyCL *cl;
    LayerDimensions dim;
//    CLKernel *broadcastMultiply;
//    CLKernel *applyActivationDeriv;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    VIRTUAL ~BackwardGpuNaive();
    VIRTUAL void backward(int batchSize,
    CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
    CLWrapper *gradInputWrapper);
    BackwardGpuNaive(EasyCL *cl, LayerDimensions dim);
    void setupBuilderBackward(TemplatedKernel *builder);
    void buildKernelBackward( string kernelSource);
    void inferenceBackward(string& kernelSource);
    void  setActivationFunction(TemplatedKernel *builder);



    // [[[end]]]
};

