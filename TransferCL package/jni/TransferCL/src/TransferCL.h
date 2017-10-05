// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

// convenience header, to include what we need, without causing whole world to rebuild
// at the same time :-) (cf, if we put in NeuralNet.h)

#include "dependencies.h"
#include "../EasyCL/EasyCL.h"

#include "netdef/NetdefToNet.h"
#include "net/Trainable.h"
#include "net/NeuralNet.h"


#include "trainers/Trainer.h"
#include "trainers/SGD.h"

#include "weights/UniformInitializer.h"
#include "weights/OriginalInitializer.h"

#include "normalize/NormalizationHelper.h"
#include "layer/Layer.h"
#include "conv/ConvolutionalLayer.h"
#include "input/InputLayer.h"
#include "layer/LayerMakers.h"

#include "batch/BatchProcess.h"
#include "batch/NetLearner.h"
#include "batch/NetLearnerOnDemand.h"
#include "batch/NetLearnerOnDemandv2.h"

#include "weights/WeightsPersister.h"
#include "util/FileHelper.h"
#include "loaders/GenericLoader.h"
#include "loaders/GenericLoaderv2.h"


#include "TransferCLDllExport.h"


#define STATIC static
#define VIRTUAL virtual

class TransferCL_EXPORT TransferCL : public EasyCL {
public:
//    EasyCL *cl;
    //ClBlasInstance clBlasInstance;
    
    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    TransferCL(cl_platform_id platformId, cl_device_id deviceId);
    ~TransferCL();
    void deleteMe();
    STATIC TransferCL *createForFirstGpu();
    STATIC TransferCL *createForFirstGpuOtherwiseCpu();
    STATIC TransferCL *createForIndexedDevice(int device);
    STATIC TransferCL *createForIndexedGpu(int gpu);
    STATIC TransferCL *createForPlatformDeviceIndexes(int platformIndex, int deviceIndex);
    STATIC TransferCL *createForPlatformDeviceIds(cl_platform_id platformId, cl_device_id deviceId);

    // [[[end]]]
};

