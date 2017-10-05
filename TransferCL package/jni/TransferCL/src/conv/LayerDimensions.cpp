// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include <iostream>
#include <ostream>

#include "../util/stringhelper.h"

#include "LayerDimensions.h"

using namespace std;

ostream &operator<<(ostream &os, const LayerDimensions &dim) {
    os << "LayerDimensions{";
    os << " inputPlanes=" << dim.inputPlanes;
    os << " inputSize=" << dim.inputSize;
    os << " numFilters=" << dim.numFilters;
    os << " filterSize=" << dim.filterSize;
    os << " outputSize=" << dim.outputSize;
    os << " padZeros=" << dim.padZeros;
    os << " biased=" << dim.biased;
    os << " skip=" << dim.skip;
    os << "}";
    return os;
}

void LayerDimensions::deriveOthers() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/LayerDimensions.cpp: deriveOthers");
#endif



	this->stride = stride;
	this->skip=stride-1;

    this->numInputPlanes = inputPlanes;
    this->isEven = filterSize % 2 == 0;
    this->outputSize = padZeros ? 
            (filterSize % 2 == 0 ? inputSize / (skip + 1) + 1 : inputSize / (skip + 1) ) :
            (inputSize - filterSize) / (skip + 1) + 1;

    this->inputSizeSquared = inputSize * inputSize;
    this->filterSizeSquared = filterSize * filterSize;
    this->outputSizeSquared = outputSize * outputSize;

    this->inputCubeSize = inputPlanes * inputSizeSquared;
    this->filtersSize = inputPlanes * numFilters * filterSizeSquared;
    this->outputCubeSize = numFilters * outputSizeSquared;

    this->halfFilterSize = filterSize >> 1;

//    this->useMaxPooling=true;

    if (useMaxPooling){
//		this->maxPool_spatialExtent=2;
//		this->maxPool_strides=2;
		this->maxPool_sizeOutput=(outputSize-maxPool_spatialExtent)/maxPool_strides + 1;
    }
    //this->useHalfMemory=true;
//    cout << "deriveOthers()" << *this << endl;
}

string LayerDimensions::buildOptionsString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/LayerDimensions.cpp: buildOptionsString");
#endif


    string options = "";
    if(biased) {
         options += " -D BIASED";
    }
    options += " -D gNumInputPlanes=" + toString(inputPlanes);
    options += " -D gInputPlanes=" + toString(inputPlanes);
    options += " -D gInputSize=" + toString(inputSize);
    options += " -D gInputSizeSquared=" + toString(square(inputSize));
    options += " -D gNumFilters=" + toString(numFilters);
    options += " -D gFilterSize=" + toString(filterSize);
    options += " -D gHalfFilterSize=" + toString(filterSize >> 1);
    options += " -D gFilterSizeSquared=" + toString(square(filterSize));
    options += " -D gNumOutputPlanes=" + toString(numFilters);
    options += " -D gOutputPlanes=" + toString(numFilters);
    options += " -D gOutputSize=" + toString(outputSize);
    options += " -D gOutputSizeSquared=" + toString(square(outputSize));
    options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);
    options += " -D gMargin=" + toString(padZeros ? filterSize >> 1 : 0);
    options += " -D gEven=" + toString(filterSize % 2 == 0 ? 1 : 0);
    options += " -D gSkip=" + toString(skip);
    return options;
}
