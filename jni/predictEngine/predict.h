//
//  openCLNR.h
//  OpenCL Example1
//
//  Created by Rasmusson, Jim on 18/03/13.
//
//  Copyright (c) 2013, Sony Mobile Communications AB
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Sony Mobile Communications AB nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
	PredictionModel();
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
