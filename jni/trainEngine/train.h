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

#ifndef TRAIN_H
#define TRAIN_H

using namespace std;

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <ctime>

#include "TransferCL/EasyCL/EasyCL.h"
#include "TransferCL/EasyCL/CLKernel.h"
#include <boost/iostreams/device/mapped_file.hpp>



#define CL_USE_DEPRECATED_OPENCL_1_1_APIS


#include "sonyOpenCLexample1.h"

class ConfigTraining {
public:

	string memory_map_file_label;
	int nbLabelData;
	string memory_map_file_data;
	string normalizationfile;
	int memMapFileTotalSize;
	int numPlanes;
	int imageSize;
    int gpuIndex;
    string dataDir;
    string trainFile;
    string dataset;
    string validateFile;
    int numTrain;
    int numTest;
    int batchSize;
    int numEpochs;
    string netDef;
    int loadWeights;
    string weightsFile;
    string weightsStoreFile;
    float writeWeightsInterval;
    string normalization;
    float normalizationNumStds;
    int dumpTimings;
    int multiNet;
    int loadOnDemand;
    int fileReadBatches;
    int normalizationExamples;
    string weightsInitializer;
    float initialWeights;
    string trainer;
    float learningRate;
    float rho;
    float momentum;
    float weightDecay;
    float anneal;


    ConfigTraining();

    string getTrainingString();

    string getOldTrainingString();
};

class TrainModel{
public:
	EasyCL *cl;
	TrainModel();
	void go(ConfigTraining config);
	void printUsage(char *argv[], ConfigTraining config);
	int trainCmd(std::string argument);
	int prepareConfig(int parameterNb, char *argList[]);
	int prepareFiles(string pathOriginalFile,int trainingExample, int inputChannel,string pathNewFileData,string pathNewFileLabel, string pathNormalizationFile);
	void  data_normalization(float * trainData,int Ntrain,float* translate, float* scale, int inputCubeSize );
	void  compare_two_binary_files(FILE *fp1, FILE *fp2);
	~TrainModel();
};

#endif
