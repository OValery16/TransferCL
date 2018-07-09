// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <string>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "../util/FileHelper.h"
#include "../util/stringhelper.h"
#include "ManifestLoaderv1.h"
#include "../util/JpegHelper.h"

#include "../TransferCLDllExport.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC STATIC bool ManifestLoaderv1::isFormatFor(std::string imagesFilepath) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: isFormatFor");
#endif


	//LOGI( "ManifestLoaderv1 checking format for %s", imagesFilepath.c_str());
    char *headerBytes = FileHelper::readBinaryChunk(imagesFilepath, 0, 1024);
    string sigString = "# format=deepcl-jpeg-list-v1 ";
    headerBytes[sigString.length()] = 0;
    bool matched = string(headerBytes) == sigString;
    cout << "matched: " << matched << endl;
    //LOGI( "matched %d",matched);
    delete[] headerBytes;
    return matched;
}

//PoolingLayer::~PoolingLayer() {
PUBLIC ManifestLoaderv1::~ManifestLoaderv1(){
	delete[] files;
	delete[] labels;
}

PUBLIC ManifestLoaderv1::ManifestLoaderv1(std::string imagesFilepath) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: ManifestLoaderv1 1");
#endif


    init(imagesFilepath, true);    
}
PUBLIC ManifestLoaderv1::ManifestLoaderv1(std::string imagesFilepath, bool includeLabels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: ManifestLoaderv1 2");
#endif


    init(imagesFilepath, includeLabels);
}
PRIVATE void ManifestLoaderv1::init(std::string imagesFilepath, bool includeLabels) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: init");
#endif


    this->includeLabels = includeLabels;
    this->imagesFilepath = imagesFilepath;
    // by reading the number of lines in the manifest, we can get the number of examples, *p_N
    // number of planes is .... 1
    // imageSize is ...

    if(!isFormatFor(imagesFilepath) ) {
    	LOGE( "file %s is not a deepcl-jpeg-list-v1 manifest file",imagesFilepath.c_str());
        throw runtime_error("file " + imagesFilepath + " is not a deepcl-jpeg-list-v1 manifest file");
    }

    ifstream infile(imagesFilepath);
    char lineChars[1024];
    infile.getline(lineChars, 1024); // skip first, header, line
    string firstLine = string(lineChars);
//    cout << "firstline: [" << firstLine << "]" << endl;
    vector<string> splitLine = split(firstLine, " ");
    N = readIntValue(splitLine, "N");
    planes = readIntValue(splitLine, "planes");
    size = readIntValue(splitLine, "width");
    int imageSizeRepeated = readIntValue(splitLine, "height");
    if(size != imageSizeRepeated) {
    	LOGE( "file %s contains non-square images.  Not handled for now.",imagesFilepath.c_str());
        throw runtime_error("file " + imagesFilepath + " contains non-square images.  Not handled for now.");
    }
    // now we should load into memory, since the file is not fixed-size records, and cannot be loaded partially easily

    files = new string[N];
    labels = new int[N];

    int n = 0;
    while(infile) {
        infile.getline(lineChars, 1024);
        if(!infile) {
            break;
        }
        string line = string(lineChars);
        if(line == "") {
            continue;
        }
        vector<string> splitLine = split(line, " ");
        if((int)splitLine.size() == 0) {
            continue;
        }
        if(includeLabels && (int)splitLine.size() != 2) { 
        	LOGI("Error reading %s.  Following line not parseable: %s",imagesFilepath.c_str(),line.c_str());
            throw runtime_error("Error reading " + imagesFilepath + ".  Following line not parseable:\n" + line);
        }
        string jpegFile = splitLine[0];
        files[n] = jpegFile;
        if(includeLabels) {
            int label = atoi(splitLine[1]);
        labels[n] = label;
        }
//        cout << "file " << jpegFile << " label=" << label << endl;
        n++;
    }
    infile.close();
    cout << "manifest " << imagesFilepath << " read. N=" << N << " planes=" << planes << " size=" << size << endl;
}
PUBLIC VIRTUAL std::string ManifestLoaderv1::getType() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: string ManifestLoaderv1::getType");
#endif


    return "ManifestLoaderv1";
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageCubeSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: getImageCubeSize");
#endif


    return planes * size * size;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: getN");
#endif


    return N;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getPlanes() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: getPlanes");
#endif


    return planes;
}
PUBLIC VIRTUAL int ManifestLoaderv1::getImageSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: getImageSize");
#endif


    return size;
}
int ManifestLoaderv1::readIntValue(std::vector< std::string > splitLine, std::string key) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: readIntValue");
#endif


    for(int i = 0; i < (int)splitLine.size(); i++) {
        vector<string> splitPair = split(splitLine[i], "=");
        if((int)splitPair.size() == 2) {
            if(splitPair[0] == key) {
                return atoi(splitPair[1]);
            }
        }
    }
    throw runtime_error("Key " + key + " not found in file header");
}
PUBLIC VIRTUAL void ManifestLoaderv1::load(unsigned char *data, int *labels, int startRecord, int numRecords) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/ManifestLoaderv1.cpp: load");
#endif


    int imageCubeSize = planes * size * size;
//    cout << "ManifestLoaderv1, loading " << numRecords << " jpegs" << endl;

    //olivier added multicor usage
    //#pragma omp parallel for
    //
    for(int localN = 0; localN < numRecords; localN++) {
        int globalN = localN + startRecord;
//        LOGI("[%s]",files[globalN].c_str());
        JpegHelper::read(files[globalN], planes, size, size, data + localN * imageCubeSize);
        if(labels != 0) {
            if(!includeLabels) {
            	LOGE( "---------------------DeepCL/src/loaders/ManifestLoaderv1.cpp data corrupted");
            	LOGE( "---------------------DeepCL/src/loaders/ManifestLoaderv1.cpp ManifestLoaderv1: labels reqested in load() method, but not activated in constructor");
                //throw runtime_error("ManifestLoaderv1: labels reqested in load() method, but not activated in constructor");
            }
            labels[localN] = this->labels[globalN];
        }

    }
}

