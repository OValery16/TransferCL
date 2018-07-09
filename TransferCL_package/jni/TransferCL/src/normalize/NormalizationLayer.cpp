// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "NormalizationLayerMaker.h"

#include "NormalizationLayer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 

NormalizationLayer::NormalizationLayer(Layer *previousLayer, NormalizationLayerMaker *maker) :
       Layer(previousLayer, maker),
    translate(maker->_translate),
    scale(maker->_scale),
    outputPlanes(previousLayer->getOutputPlanes()),
    outputSize(previousLayer->getOutputSize()),
    batchSize(maker->_batchsize),
    allocatedSize(0),
    output(0),
    cl(maker->cl),
	outputWrapper(0){
	//forwardNorm=new clNormalization(maker->cl,translate,scale, getOutputNumElements(),use_Half);
}
VIRTUAL NormalizationLayer::~NormalizationLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: ~NormalizationLayer");
#endif




}
VIRTUAL std::string NormalizationLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: string NormalizationLayer::getClassName");
#endif


    return "NormalizationLayer";
}
VIRTUAL float *NormalizationLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutput");
#endif


    return output;
}
VIRTUAL ActivationFunction const *NormalizationLayer::getActivationFunction() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getActivationFunction");
#endif


    return new LinearActivation();
}
VIRTUAL int NormalizationLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getPersistSize");
#endif


    if(version == 1) {
        return 0;
    }
    return 2;
}
VIRTUAL void NormalizationLayer::persistToArray(int version, float *array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: persistToArray");
#endif


    if(version == 1) {
        return;
    }
    array[0] = translate;
    array[1] = scale;

#if DISPLAY_WEIGHT ==1
    LOGI("######################################");
    LOGI("######## NormalizationLayer s weights #############");
    LOGI("######################################");

        LOGI("translate %f scale %f", array[0],array[1]);
#endif
}
/// \brief initialize the current weights and biases from array
VIRTUAL void NormalizationLayer::unpersistFromArray(int version, float const*array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: unpersistFromArray");
#endif

#if DISPLAY_WEIGHT ==1
LOGI("######################################");
LOGI("######## NormalizationLayer s weights #############");
LOGI("######################################");

    LOGI("translate %f scale %f", array[0],array[1]);
#endif
    if(version == 1) {
        return;
    }
    translate = array[0];
    scale = array[1];
}
VIRTUAL bool NormalizationLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: needsBackProp");
#endif


    return previousLayer->needsBackProp();
}
VIRTUAL void NormalizationLayer::printOutput() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: printOutput");
#endif


    if(output == 0) {
         return;
    }
    for(int n = 0; n < std::min(5,batchSize); n++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: min(5,batchSize); n++) {");
#endif


        std::cout << "NormalizationLayer n " << n << ":" << std::endl;
        for(int plane = 0; plane < std::min(5, outputPlanes); plane++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: min(5, outputPlanes); plane++) {");
#endif


            if(outputPlanes > 1) std::cout << "    plane " << plane << ":" << std::endl;
            for(int i = 0; i < std::min(5, outputSize); i++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: min(5, outputSize); i++) {");
#endif


                std::cout << "      ";
                for(int j = 0; j < std::min(5, outputSize); j++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: min(5, outputSize); j++) {");
#endif


                    std::cout << getResult(n, plane, i, j) << " ";
//output[
//                            n * numPlanes * imageSize*imageSize +
//                            plane*imageSize*imageSize +
//                            i * imageSize +
//                            j ] << " ";
                }
                if(outputSize > 5) std::cout << " ... ";
                std::cout << std::endl;
            }
            if(outputSize > 5) std::cout << " ... " << std::endl;
        }
        if(outputPlanes > 5) std::cout << "   ... other planes ... " << std::endl;
    }
    if(batchSize > 5) std::cout << "   ... other n ... " << std::endl;
}
VIRTUAL void NormalizationLayer::print() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: print");
#endif


    printOutput();
}
VIRTUAL bool NormalizationLayer::needErrorsBackprop() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: needErrorsBackprop");
#endif


    return false;
}
VIRTUAL void NormalizationLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
//    if(output != 0) {
//        delete[] output;
//    }
    this->batchSize = batchSize;
    this->allocatedSize = allocatedSize;
//    output = new float[ getOutputNumElements() ];
}
VIRTUAL void NormalizationLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: forward");
#endif

outputWrapper=previousLayer->getOutputWrapper();
output=previousLayer->getOutput();
//    int totalLinearLength = getOutputNumElements();
//    float* input = previousLayer->getOutput();
//    for(int i = 0; i < totalLinearLength; i++) {
//        output[i] = /*(*/input[i]/* + translate) * scale*/;
//    }

//    if (use_Half){
//    	outputh=new half[totalLinearLength];
//    	for(int i = 0; i < totalLinearLength; i++) {
//    		outputh[i]=FloatToHalf(output[i]);
//    	}
//    	CLWrapper * testWrapper1 = (CLWrapper *)cl->wrap(totalLinearLength,outputh);
//    	testWrapper1->copyToDevice();
//    	forwardNorm->NormalizationZeroMeanHalf(testWrapper1,totalLinearLength);
//    	LOGI("I m done");
//    	testWrapper1->copyToHost();
//    	float sum =0.0f;
//        for(int i = 0; i < totalLinearLength; i++) {
//    	sum =sum + abs( HalfToFloat(outputh[i])-((output[i] + translate) * scale));
//        }
//    	delete testWrapper1;
//    	delete [] outputh;
//
//    	    for(int i = 0; i < totalLinearLength; i++) {
//    	        output[i] = (output[i] + translate) * scale;
//    	    }
//    	outputWrapper = (CLWrapper *)cl->wrap(totalLinearLength, output);
//    	outputWrapper->copyToDevice();
//    }else{
//    	if (outputWrapper==0)
//    		outputWrapper = (CLWrapper *)cl->wrap(totalLinearLength, output);
//
//
//    	outputWrapper->copyToDevice();
//    	forwardNorm->NormalizationZeroMean(outputWrapper,totalLinearLength);
////    	outputWrapper->copyToHost();
////    	float sum =0.0f;
////    	    for(int i = 0; i < totalLinearLength; i++) {
////    	sum =sum + abs( upstreamOutput[i]-output[i]);
////    	    }
////    	    LOGI("sum norm %f",sum);
////    	delete [] copytest;
//    }
}
VIRTUAL bool NormalizationLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: hasOutputWrapper");
#endif


    return true;//false;
}
VIRTUAL CLWrapper *NormalizationLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}
VIRTUAL void NormalizationLayer::backward(float learningRate, float const *gradOutput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: backward");
#endif


  // do nothing...
}
VIRTUAL int NormalizationLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL bool NormalizationLayer::isFirstLayer() const {
   #if TRANSFERCL_VERBOSE == 1
   LOGI( "DeepCL/src/layer/Layer.cpp: isFirstLayer");
   #endif


       return true;
   }
VIRTUAL float NormalizationLayer::getTranslate() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputPlanes");
#endif


    return translate;
}

VIRTUAL float NormalizationLayer::getScale() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputPlanes");
#endif


    return scale;
}

VIRTUAL int NormalizationLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputPlanes");
#endif


    return outputPlanes;
}
VIRTUAL int NormalizationLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputCubeSize");
#endif


    return outputPlanes * outputSize * outputSize;
}
VIRTUAL int NormalizationLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: getOutputNumElements");
#endif


    return batchSize * getOutputCubeSize();
}
VIRTUAL std::string NormalizationLayer::toString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: string NormalizationLayer::toString");
#endif


    return toString();
}
VIRTUAL std::string NormalizationLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/normalize/NormalizationLayer.cpp: string NormalizationLayer::asString");
#endif


    return std::string("") + "NormalizationLayer{ outputPlanes=" + ::toString(outputPlanes) + " outputSize=" +  ::toString(outputSize) + " translate=" + ::toString(translate) + " scale=" + ::toString(scale) + " }";
}


