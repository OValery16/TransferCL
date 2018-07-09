// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#ifndef PREDICT_H
#define PREDICT_H

using namespace std;

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <unistd.h>

#include "TransferCL/EasyCL/EasyCL.h"
#include "TransferCL/EasyCL/CLKernel.h"

//#include "../DeepCL/src/clblas/ClBlasInstance.h"
#include "../TransferCL/src/TransferCL.h"
#include "../TransferCL/src/loss/SoftMaxLayer.h"
#include "../TransferCL/src/loss/SoftMaxLayerPredict.h"

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS


#include "sonyOpenCLexample1.h"

class ConfigPrediction {
public:

    int gpuIndex;
    string weightsFile;
    int batchSize;
    string inputFile;
    string outputFile;
    int outputLayer;
    int writeLabels;
    string outputFormat;


    ConfigPrediction();

};

class PredictionModel{
public:
	EasyCL *cl;
	PredictionModel(string absolutePath);
	void makePredictions(int n,NeuralNet *net,float *inputData,ConfigPrediction config,ostream *outFile,int *labels,const long inputCubeSize, int N, GenericLoaderv2* loader);
	void go(ConfigPrediction config);
	void go0(ConfigPrediction config);
	void printUsage(char *argv[], ConfigPrediction config);
	int predictCmd(std::string argument);
	int prepareConfig(int parameterNb, char *argList[]);
	int WormUpGPU(ConfigPrediction config);
	~PredictionModel();
};

#endif
