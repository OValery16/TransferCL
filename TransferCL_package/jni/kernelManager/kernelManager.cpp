// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
#include "../DeepCL/src/util/stringhelper.h"

#include "kernelManager.h"

using namespace std;

KernelManager::KernelManager(std::string fileDirectory,string binaryFilesRepo){


	congigurationManager=new ConfigManager(fileDirectory,binaryFilesRepo);
	cl = EasyCL::createForFirstGpuOtherwiseCpu(congigurationManager);

	string filepath="";
	string line;
	ifstream myfileI (fileDirectory);
	if (myfileI.is_open())
	{
	   while ( getline (myfileI,line) )
	   {
		   vector<string> splitData=split(line, ",");
		   listOfCompiledKernel.insert ({splitData[0],splitData[1]});
        }
    myfileI.close();
    }
	kernellList=fileDirectory;
	binaryRepo=binaryFilesRepo;

//	for (auto& x: listOfCompiledKernel)
//	    LOGI("%s",x.first.c_str()/*, x.second.c_str()*/);


}

void KernelManager::CompileKernels(string fileDirectory){

	vector<string> listFernel=loadListKernelToCompile(fileDirectory);

	for (auto& config: listFernel){

		vector<string> parametersList=split(config, " ");
		CompileKernel(parametersList[0] ,parametersList[1] ,atoi(parametersList[2]),atoi(parametersList[3]));
	}

}

void KernelManager::CompileKernel(string batchsizeString,string netDef,int numPlanes,int imageSize){

	//string netDef="8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-2n";

	//LOGE("batchsize %s",batchsizeString.c_str());
	int batchsize = atoi(batchsizeString.substr(batchsizeString.find("=")+1));
	//LOGE("batchsize %d",batchsize);
	//LOGE("netDef %s",netDef.c_str());

	//int batchsize = netDef
	NeuralNet *net;
	net = new NeuralNet(cl);
	net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
	net->addLayer(NormalizationLayerMaker::instance()->translate(0.0f)->scale(1.0f) ); // This will be read from weights file
	WeightsInitializer *weightsInitializer = new OriginalInitializer();
	//NetdefToCompile *ndtc=new NetdefToCompile();
	//ndtc->createNetForCompile(net, netDef, weightsInitializer);
	//delete ndtc;
	NetdefToCompile::createNetForCompile(batchsize, net, netDef, weightsInitializer);
	//NetdefToNet::createNetFromNetdef(net, netDef, weightsInitializer);
	delete net;
}

KernelManager::~KernelManager(){

	LOGI( "KernelManager destroyed");
	delete cl;
	delete congigurationManager;

}

vector<string> KernelManager::loadListKernelToCompile(std::string fileDirectory){

	vector<string> listOfKernelToCompile;
	string filepath="";
	string line;
	ifstream myfileI (fileDirectory);
	if (myfileI.is_open())
	{
	   while ( getline (myfileI,line) )
	   {
		   listOfKernelToCompile.push_back(line);
	   }
	myfileI.close();
	}

//	for (auto& x: listOfKernelToCompile)
//       LOGI("%s", x.c_str());
	return listOfKernelToCompile;

}

bool KernelManager::SaveProgramBinary(cl_program program, cl_device_id device, const char* fileName)
{
    cl_uint numDevices = 0;
    cl_int errNum;

    // 1 - Query for number of devices attached to program
    errNum = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(cl_uint),
                              &numDevices, NULL);
    if (errNum != CL_SUCCESS)
    {
        LOGE( "Error querying for number of devices." );
        return false;
    }

    // 2 - Get all of the Device IDs
    cl_device_id *devices = new cl_device_id[numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_DEVICES,
                              sizeof(cl_device_id) * numDevices,
                              devices, NULL);
    if (errNum != CL_SUCCESS)
    {
    	LOGE( "Error querying for devices.");
        delete [] devices;
        return false;
    }

    // 3 - Determine the size of each program binary
    size_t *programBinarySizes = new size_t [numDevices];
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                              sizeof(size_t) * numDevices,
                              programBinarySizes, NULL);
    if (errNum != CL_SUCCESS)
    {
    	LOGE( "Error querying for program binary sizes.");
        delete [] devices;
        delete [] programBinarySizes;
        return false;
    }

    unsigned char **programBinaries = new unsigned char*[numDevices];
    for (cl_uint i = 0; i < numDevices; i++)
    {
        programBinaries[i] = new unsigned char[programBinarySizes[i]];
    }

    // 4 - Get all of the program binaries
    errNum = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*) * numDevices,
                              programBinaries, NULL);
    if (errNum != CL_SUCCESS)
    {
    	 LOGE( "Error querying for program binaries");

        delete [] devices;
        delete [] programBinarySizes;
        for (cl_uint i = 0; i < numDevices; i++)
        {
            delete [] programBinaries[i];
        }
        delete [] programBinaries;
        return false;
    }

    // 5 - Finally store the binaries for the device requested out to disk for future reading.
    for (cl_uint i = 0; i < numDevices; i++)
    {
        // Store the binary just for the device requested.  In a scenario where
        // multiple devices were being used you would save all of the binaries out here.
        if (devices[i] == device)
        {
            FILE *fp = fopen(fileName, "wb");
            fwrite(programBinaries[i], 1, programBinarySizes[i], fp);
            fclose(fp);
            break;
        }
    }

    // Cleanup
    delete [] devices;
    delete [] programBinarySizes;
    for (cl_uint i = 0; i < numDevices; i++)
    {
        delete [] programBinaries[i];
    }
    delete [] programBinaries;
    return true;
}

bool KernelManager::alreadyCompiledKernel(string kernelname, string option,string &filepath){

	string key=kernelname+" "+option;
	std::unordered_map<string,string>::const_iterator got = listOfCompiledKernel.find (key);

	if ( got == listOfCompiledKernel.end() ){
	    LOGI("not found in the list");
	    LOGI("find the name of the binary file such as name_number.bin");
	    int i=0;
	    for (auto& x: listOfCompiledKernel) {
	    	if (x.first.find(kernelname)!=std::string::npos){
        		i++;
        	}
	    	string kernelname2=kernelname+"_"+std::to_string(i);
	    	string filepath=binaryRepo+kernelname2+".bin";
	        //std::cout << x.first << ": " << x.second << std::endl;
	      }
	    return false;
	}else{
		  //LOGI("%s %s", got->first.c_str() , got->second.c_str());
		  filepath=got->second.c_str();
		  return true;
	  }

	return true;
}

void KernelManager::addNewCompiledKernel(string kernelname, string options,string &filepath){

	string key=kernelname+" "+options;
	listOfCompiledKernel.insert ({key,filepath});

	ofstream myfileO;
	myfileO.open (kernellList,std::ofstream::out | std::ofstream::app);
	myfileO <<kernelname<<" " <<options <<","<<filepath<<"\n";
	myfileO.close();
}




