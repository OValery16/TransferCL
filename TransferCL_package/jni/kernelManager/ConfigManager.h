// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifndef TESTLIST_H
#define TESTLIST_H

using namespace std;

#include <stdio.h>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <fstream>


#include "../TransferCL/src/dependencies.h"

class ConfigManager{
public:
	std::unordered_map<std::string,std::string> listOfCompiledKernel;//configuration <--> kernel path
	bool alreadyCompiledKernel(string kernelname, string option,string operation,string &filepath);
	void addNewCompiledKernel(string kernelname, string options,string operation,string &filepath);
	//ConfigManager(std::string fileDirectory,string binaryFilesRepo);
	ConfigManager(std::string fileDirectory);
	//ConfigManager();
private:


	std::string binaryRepo="";///data/data/com.sony.openclexample1/directoryTest/binariesKernel/";
	std::string kernellList="";//"/data/data/com.sony.openclexample1/directoryTest/binariesKernel/list.txt";

};

#endif
