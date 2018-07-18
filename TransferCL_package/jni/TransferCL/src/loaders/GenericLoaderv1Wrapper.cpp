// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "../TransferCLDllExport.h"

#include "GenericLoader.h"
#include "GenericLoaderv1Wrapper.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

PUBLIC VIRTUAL std::string GenericLoaderv1Wrapper::getType() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: string GenericLoaderv1Wrapper::getType");
#endif


    return "GenericLoaderv1Wrapper";
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getN() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: getN");
#endif


    return N;
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getPlanes() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: getPlanes");
#endif


    return planes;
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getImageSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: getImageSize");
#endif


    return size;
}
PUBLIC GenericLoaderv1Wrapper::GenericLoaderv1Wrapper(std::string imagesFilepath) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: GenericLoaderv1Wrapper");
#endif


    this->imagesFilepath = imagesFilepath;
    GenericLoader::getDimensions(imagesFilepath.c_str(), &N, &planes, &size);
}
PUBLIC VIRTUAL int GenericLoaderv1Wrapper::getImageCubeSize() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: getImageCubeSize");
#endif


    return planes * size * size;
}
PUBLIC VIRTUAL void GenericLoaderv1Wrapper::load(unsigned char *data, int *labels, int startRecord, int numRecords) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/loaders/GenericLoaderv1Wrapper.cpp: load");
#endif


    GenericLoader::load(imagesFilepath.c_str(), data, labels, startRecord, numRecords);
}

