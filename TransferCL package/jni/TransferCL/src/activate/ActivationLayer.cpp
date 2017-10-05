// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include "../net/NeuralNet.h"
#include "../util/stringhelper.h"

#include "ActivationLayer.h"
#include "ActivationMaker.h"
//#include "ActivationForward.h"
//#include "ActivationBackward.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL
#undef STATIC
#define STATIC

ActivationLayer::ActivationLayer(EasyCL *cl, Layer *previousLayer, ActivationMaker *maker) :
        Layer(previousLayer, maker),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        outputSize(previousLayer->getOutputSize()),
        fn(maker->_activationFunction),
        cl(cl),
        output(0),
        gradInput(0),
        outputWrapper(0),
        gradInputWrapper(0),
        setup(false),
//        outputCopiedToHost(false),
//        gradInputCopiedToHost(false),
        batchSize(0),
        allocatedSize(0) {
    if(inputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
        throw runtime_error("Error: Activation layer " + toString(layerIndex) + ": output image size is 0");
    }
//    activationForwardImpl = ActivationForward::instance(cl, numPlanes, inputSize, fn);
//    activationBackpropImpl = ActivationBackward::instance(cl, numPlanes, inputSize, fn);
}
VIRTUAL ActivationLayer::~ActivationLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: ~ActivationLayer");
#endif

}
VIRTUAL std::string ActivationLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: string ActivationLayer::getClassName");
#endif


    return "ActivationLayer";
}
VIRTUAL float ActivationLayer::getOutput(int n, int plane, int row, int col) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutput");
#endif


    int index = (( n
        * numPlanes + plane)
        * outputSize + row)
        * outputSize + col;
    return output[ index ];
}
VIRTUAL void ActivationLayer::printOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: printOutput");
#endif


//    float const*output = getOutput();
//    int outPlanes = getOutputPlanes();
//    int outputNumElements = getOutputSize();
    //std::cout << "  outputs: " << std::endl;
    getOutput();
// output are organized like [imageid][filterid][row][col]
    for(int n = 0; n < std::min(5, batchSize); n++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: min(5, batchSize); n++) {");
#endif


        std::cout << "    n: " << n << std::endl;
        for(int plane = 0; plane < std::min(5, numPlanes); plane++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: min(5, numPlanes); plane++) {");
#endif


            if(numPlanes > 1) std::cout << "      plane " << plane << std::endl;
            if(outputSize == 1) {
                 std::cout << "        " << getOutput(n, plane, 0, 0) << std::endl;
            } else {
                for(int i = 0; i < std::min(5, outputSize); i++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: min(5, outputSize); i++) {");
#endif


                    std::cout << "      ";
                    for(int j = 0; j < std::min(5, outputSize); j++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: min(5, outputSize); j++) {");
#endif


                        std::cout << getOutput(n, plane, i, j) << " ";
                    }
                    if(outputSize > 5) std::cout << " ... ";
                    std::cout << std::endl;
                }
                if(outputSize > 5) std::cout << " ... " << std::endl;
            }
            if(numPlanes > 5) std::cout << " ... other planes ... " << std::endl;
        }
        if(batchSize > 5) std::cout << " ... other n ... " << std::endl;
    }
}
VIRTUAL void ActivationLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: setBatchSize");
#endif


//    cout << "ActivationLayer::setBatchSize" << endl;
    if(batchSize <= allocatedSize) {
        this->batchSize = batchSize;
        return;
    }
//    if(outputWrapper != 0) {
//        delete outputWrapper;
//    }
//    if(output != 0) {
//        delete[] output;
//    }
//    if(gradInputWrapper != 0) {
//        delete gradInputWrapper;
//    }
//    if(gradInput != 0) {
//        delete[] gradInput;
//    }
    if (not setup){
		this->batchSize = batchSize;
		this->allocatedSize = batchSize;
		output = 0;
		outputWrapper = 0;
		gradInput = 0;
		gradInputWrapper = 0;
		setup=true;
    }
//    output = new float[ getOutputNumElements() ];
//    outputWrapper = cl->wrap(getOutputNumElements(), output);
//    outputWrapper->createOnDevice();
//    gradInput = new float[ previousLayer->getOutputNumElements() ];
//    gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), gradInput);
//    gradInputWrapper->createOnDevice();
}
VIRTUAL int ActivationLayer::getOutputNumElements() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *ActivationLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutput");
#endif


    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
//    cout << "getOutput output[0] " << output[0] << " output[1] " << output[1] << endl;
    return output;
}
VIRTUAL bool ActivationLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: needsBackProp");
#endif


    return previousLayer->needsBackProp();
}
VIRTUAL int ActivationLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputNumElements");
#endif


//    int outputSize = inputSize / poolingSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int ActivationLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputCubeSize");
#endif


    return numPlanes * outputSize * outputSize;
}
VIRTUAL int ActivationLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int ActivationLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputPlanes");
#endif


    return numPlanes;
}
VIRTUAL bool ActivationLayer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}
VIRTUAL bool ActivationLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: hasOutputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *ActivationLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}
VIRTUAL int ActivationLayer::getWeightsSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getWeightsSize");
#endif


    return 0;
}
VIRTUAL int ActivationLayer::getBiasSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getBiasSize");
#endif


    return 0;
}
VIRTUAL float *ActivationLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getGradInput");
#endif


    if(gradInputWrapper->isDeviceDirty()) {
        gradInputWrapper->copyToHost();
//        gradInputCopiedToHost = true;
    }
    return gradInput;
}
VIRTUAL ActivationFunction const *ActivationLayer::getActivationFunction() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getActivationFunction");
#endif


    return fn;
}

CLWrapper *ActivationLayer::getSelectorWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBiasWrapper");
#endif


    return selectorWrapper;
}
VIRTUAL void ActivationLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: forward");
#endif

//LOGI("isDeviceDirty %d",outputWrapper->isDeviceDirty());
outputWrapper= previousLayer->getOutputWrapper();
//LOGI("isDeviceDirty %d",outputWrapper->isDeviceDirty());
//if (typeid(previousLayer) == typeid(ConvolutionalLayer))

if (previousLayer->isConvLayer()){
	selectorWrapper= previousLayer->getSelectorWrapper();

//	int * selector0=(int*)selectorWrapper->getHostArray();
//	   LOGI("////////////selectors////////");
//					for (int i =0;i<20/*dim.outputSize/2*/;i++){
//						string displayArraY="";
//						for (int j =0;j<14;j++){
//							displayArraY= displayArraY+ "-" + to_string(selector0[i*14+j]);
//						}
//						LOGI("%s",displayArraY.c_str());
//						displayArraY.clear();
//					}
}
////
//    CLWrapper *inputWrapper = 0;
//    if(previousLayer->hasOutputWrapper()) {
//    	LOGI("activation hasOutputWrapper");
//        inputWrapper = previousLayer->getOutputWrapper();
//    } else {
//    	LOGI("activation not hasOutputWrapper");
//        float *input = previousLayer->getOutput();
//        inputWrapper = cl->wrap(previousLayer->getOutputNumElements(), input);
//        inputWrapper->copyToDevice();
//    }
//    activationForwardImpl->forward(batchSize, inputWrapper, outputWrapper);
//    LOGI("done");
//
////    outputWrapper->copyToHost();
////    float*conv=(float*)outputWrapper->getHostArray();
////    float *input = previousLayer->getOutput();
////    		float sum=0.0f;
////    		for(int i =0;i<batchSize * numPlanes * outputSize * outputSize; i++){
////    			sum+=abs(input[i]-conv[i]);
////    		}
////    		for(int i =0;i<20; i++){
////    			LOGI("%f %f",input[i],conv[i]);
////    		}
////    		LOGE("diff %f",sum);
//////    outputCopiedToHost = false;
//    if(!previousLayer->hasOutputWrapper()) {
//        delete inputWrapper;
//    }
//    LOGI("done2");
}
VIRTUAL void ActivationLayer::backward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: backward");
#endif
gradInputWrapper= nextLayer->getGradInputWrapper();

//if (not previousLayer->isConvLayer()){
//	gradInputWrapper= nextLayer->getGradInputWrapper();//olivier: done by the previous layer
//}else{
//    CLWrapper *gradOutputWrapper = 0;
//    bool weOwnGradOutputWrapper = false;
//    if(nextLayer->providesGradInputWrapper()) {
//        gradOutputWrapper = nextLayer->getGradInputWrapper();
//    } else {
//        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
//        gradOutputWrapper->copyToDevice();
//        weOwnGradOutputWrapper = true;
//    }
//    gradOutputWrapper->copyToHost();
//    float*grad=(float*)gradOutputWrapper->getHostArray();
//    for(int i= 0; i< 10; i++)
//       LOGI("grad[%d]=%f",i,grad[i]);
//
//    activationBackpropImpl->backward(batchSize, outputWrapper, gradOutputWrapper, gradInputWrapper);
////    gradInputCopiedToHost = false;
//
////    if(!previousLayer->hasOutputWrapper()) {
////        delete imagesWrapper;
////    }
//    if(weOwnGradOutputWrapper) {
//        delete gradOutputWrapper;
//    }
//}
}
VIRTUAL std::string ActivationLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: string ActivationLayer::asString");
#endif


    return std::string("ActivationLayer{ ") + fn->getDefineName() + " }";
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: string");
#endif


}
VIRTUAL int ActivationLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/activate/ActivationLayer.cpp: getPersistSize");
#endif


    // no weights, so:
    return 0;
}

