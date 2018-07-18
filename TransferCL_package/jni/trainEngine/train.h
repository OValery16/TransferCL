// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

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

	string absolutePath;
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
	TrainModel(string absolutePath);
	void go(ConfigTraining config);
	void printUsage(char *argv[], ConfigTraining config);
	int trainCmd(std::string argument,string absolutePath);
	int prepareConfig(int parameterNb, char *argList[],string absolutePath);
	int prepareFiles(string pathOriginalFile,int trainingExample, int inputChannel,string pathNewFileData,string pathNewFileLabel, string pathNormalizationFile);
	void  data_normalization(float * trainData,int Ntrain,float* translate, float* scale, int inputCubeSize );
	void  compare_two_binary_files(FILE *fp1, FILE *fp2);
	~TrainModel();
};

#endif
