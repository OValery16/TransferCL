// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

#include "../normalize/NormalizationHelper.h"

#include "../TransferCLDllExport.h"

class GenericLoaderv2;

class TransferCL_EXPORT BatchAction {
public:
    float *data;
    int *labels;
    BatchAction(float *data, int *labels) :
        data(data),
        labels(labels) { // have to provide appropriate buffers for this
    }
    virtual void processBatch(int batchSize, int cubeSize) = 0;
};

class TransferCL_EXPORT BatchProcessv2 {
public:
    static void run(GenericLoaderv2*loader, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction);
};

class TransferCL_EXPORT BatchProcess {
public:
    static void run(std::string filepath, int startN, int batchSize, int totalN, int cubeSize, BatchAction *batchAction);
};

class TransferCL_EXPORT NormalizeGetStdDev : public BatchAction {
public:
    Statistics statistics; 
    NormalizeGetStdDev(float *data, int *labels) :
        BatchAction(data, labels) {
    }
    virtual void processBatch(int batchSize, int cubeSize) {
        NormalizationHelper::updateStatistics(this->data, batchSize, cubeSize, &statistics);
    }
    void calcMeanStdDev(float *p_mean, float *p_stdDev) {
        NormalizationHelper::calcMeanAndStdDev(&statistics, p_mean, p_stdDev);
    }
};


class TransferCL_EXPORT NormalizeGetMinMax : public BatchAction {
public:
    Statistics statistics; 
    NormalizeGetMinMax(float *data, int *labels) :
        BatchAction(data, labels) {
    }
    virtual void processBatch(int batchSize, int cubeSize) {
        NormalizationHelper::updateStatistics(this->data, batchSize, cubeSize, &statistics);
    }
    void calcMinMaxTransform(float *p_translate, float *p_scale) {
        // add this to our values to center
        *p_translate = - (statistics.maxY - statistics.minY) / 2.0f;
        // multiply our values by this to scale to -1 / +1 range
        *p_scale = 1.0f / (statistics.maxY - statistics.minY);
    }
};

