// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef KERNELMANAGER_H
#define KERNELMANAGER_H

using namespace std;

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <unordered_map>

#include "NetdefToCompile.h"
#include "TransferCL/EasyCL/EasyCL.h"
#include "TransferCL/EasyCL/CLKernel.h"

//#include "../DeepCL/src/clblas/ClBlasInstance.h"
#include "../TransferCL/src/TransferCL.h"
#include "../TransferCL/src/loss/SoftMaxLayer.h"
#include "../TransferCL/src/util/stringhelper.h"
#include "ConfigManager.h"

#include <CL/cl.h>

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS


#include "sonyOpenCLexample1.h"


class KernelManager{
public:
	std::unordered_map<std::string,std::string> listOfCompiledKernel;//configuration <--> kernel path
	//std::tr1::unordered_map<std::string,std::string> listOfCompiledKernel;
	void CompileKernels(std::string fileDirectory);
	bool alreadyCompiledKernel(string kernelname, string option,string &filepath);
	void addNewCompiledKernel(string kernelname, string options,string &filepath);
	KernelManager(std::string fileDirectory,string binaryFilesRepo);
	~KernelManager();

private:
	std::string binaryRepo;
	std::string kernellList;
	EasyCL *cl;
	ConfigManager *congigurationManager;

	vector<string> loadListKernelToCompile(std::string fileDirectory);
	void CompileKernel(string batchsizeString,string netDef,int numPlanes,int imageSize);

	bool SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName);


};

#endif
