// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>

#include "../net/NeuralNet.h"
#include "../layer/Layer.h"
#include "PoolingMaker.h"
#include "PoolingLayer.h"
#include "PoolingBackward.h"

#include "../dependencies.h"

#define TEST_FORWARD_PROP 0
#define TEST_BACKWARD_PROP 0

//#include "test/PrintBuffer.h"

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

std::string to_string_with_precision8(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}

int PoolingLayer::setPreviousLayer_activationLayer(const char *_activ) {
    	if (strcmp (_activ,"LINEAR")==0)
    		return 1;
    	if (strcmp (_activ,"RELU")==0)
    		return 2;
    	if (strcmp (_activ,"TANH")==0)
    		return 3;
    	if (strcmp (_activ,"SCALEDTANH")==0)
    		return 4;
    	if (strcmp (_activ,"SIGMOID")==0)
    		return 5;
    	if (strcmp (_activ,"ELU")==0)
    		return 6;

        return -1;
    }

PoolingLayer::PoolingLayer(EasyCL *cl, Layer *previousLayer, PoolingMaker *maker) :
        Layer(previousLayer, maker),
        padZeros(maker->_padZeros),
        numPlanes (previousLayer->getOutputPlanes()),
        inputSize(previousLayer->getOutputSize()),
        poolingSize(maker->_poolingSize),
        outputSize(maker->_padZeros ? (previousLayer->getOutputSize() + maker->_poolingSize - 1) / maker->_poolingSize : previousLayer->getOutputSize() / maker->_poolingSize),
        cl(cl),
        output(0),
        selectors(0),
        gradInput(0),
        outputWrapper(0),
        selectorsWrapper(0),
        gradInputWrapper(0),
        inputWrapper(0),
        setup(false),
        batchSize(0),
        allocatedSize(0){

	bool test;
	int previousLayer_activationLayer;
	if (dynamic_cast<ActivationLayer*>(previousLayer)){
		previousLayer_activationLayer=setPreviousLayer_activationLayer((dynamic_cast<ActivationLayer*>(previousLayer))->fn->getDefineName());
	}else{
		previousLayer_activationLayer=-1;
	}
	if (dynamic_cast<ConvolutionalLayer*>(previousLayer)){
		test=(dynamic_cast<ConvolutionalLayer*>(previousLayer)->dim.test);
	}else{
		test=0;
	}


    if(inputSize == 0){
//        maker->net->print();
    	LOGI("network definition error input image size is 0");
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": input image size is 0");
    }
    if(outputSize == 0){
//        maker->net->print();
    	LOGI("network definition error output image size is 0");
        throw runtime_error("Error: Pooling layer " + toString(layerIndex) + ": output image size is 0");
    }
	#if TEST_FORWARD_PROP == 1
		poolingForwardImpl = PoolingForward::instance(cl, padZeros, numPlanes, inputSize, poolingSize);
	#endif
	#if TRANSFER==0
		poolingBackpropImpl = PoolingBackward::instance(cl, padZeros, numPlanes, inputSize, poolingSize,previousLayer_activationLayer,test);
	#endif
}
VIRTUAL PoolingLayer::~PoolingLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: ~PoolingLayer");
#endif

#if TEST_FORWARD_PROP == 1
    delete poolingForwardImpl;
#endif
#if TRANSFER==0
    delete poolingBackpropImpl;


//    if(!previousLayer->hasOutputWrapper()) {
//        delete inputWrapper;
//    }
//    if(outputWrapper != 0) {
//        delete outputWrapper;
//    }
//    if(output != 0) {
//        delete[] output;
//    }
//    if(selectorsWrapper != 0) {
//        delete selectorsWrapper;
//    }
//    if(selectors != 0) {
//        delete[] selectors;
//    }
    if(gradInputWrapper != 0) {
        delete gradInputWrapper;
    }
#endif
//    if(gradInput != 0) {
//        delete[] gradInput;
//    }
}
VIRTUAL std::string PoolingLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: string PoolingLayer::getClassName");
#endif


    return "PoolingLayer";
}
VIRTUAL void PoolingLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: setBatchSize");
#endif


//    cout << "PoolingLayer::setBatchSize" << endl;
#if TRANSFERCL_VERBOSE == 1
	LOGI("DeepCL\\src\\layer\\PoolingLayer.cpp: PoolingLayer setBatchSize");
#endif

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
//    if(selectorsWrapper != 0) {
//        delete selectorsWrapper;
//    }
//    if(selectors != 0) {
//        delete[] selectors;
//    }
//    if(gradInputWrapper != 0) {
//        delete gradInputWrapper;
//    }
//    if(gradInput != 0) {
//        delete[] gradInput;
//    }
    this->batchSize = batchSize;
    this->allocatedSize = batchSize;
	#if TEST_FORWARD_PROP == 1
		output = new float[ getOutputNumElements() ];
		outputWrapper = cl->wrap(getOutputNumElements(), output);
		selectors = new int[ getOutputNumElements() ];
		selectorsWrapper = cl->wrap(getOutputNumElements(), selectors);
	#endif
   if(not setup){
		//gradInputWrapper=(dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))->gradInput_poolingLayer_Wrapper;
		//gradInput = new float[ previousLayer->getOutputNumElements() ];
		float* temp=0;
		gradInputWrapper = cl->wrap(previousLayer->getOutputNumElements(), temp);
		gradInputWrapper->createOnDevice();
		setup=true;
	}
}
VIRTUAL int PoolingLayer::getOutputNumElements() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputNumElements");
#endif


    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL float *PoolingLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutput");
#endif


    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
//        outputCopiedToHost = true;
    }
    return output;
}
VIRTUAL bool PoolingLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: needsBackProp");
#endif


    return previousLayer->needsBackProp();
}
VIRTUAL int PoolingLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputNumElements");
#endif


//    int outputSize = inputSize / poolingSize;
    return batchSize * numPlanes * outputSize * outputSize;
}
VIRTUAL int PoolingLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputSize");
#endif


    return outputSize;
}
VIRTUAL int PoolingLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputCubeSize");
#endif


    return numPlanes * outputSize * outputSize;
}
VIRTUAL int PoolingLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputPlanes");
#endif


    return numPlanes;
}
VIRTUAL int PoolingLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getPersistSize");
#endif


    return 0;
}
VIRTUAL bool PoolingLayer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}
VIRTUAL bool PoolingLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: hasOutputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *PoolingLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}
VIRTUAL float *PoolingLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getGradInput");
#endif


    return gradInput;
}
VIRTUAL ActivationFunction const *PoolingLayer::getActivationFunction() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: getActivationFunction");
#endif


    //return previousLayer->getActivationFunction(); // I guess???
    return new LinearActivation();
}
VIRTUAL void PoolingLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: forward");
#endif


#if TEST_FORWARD_PROP == 1
	//CLWrapper *upstreamOutputWrapper = 0;
	if(previousLayer->hasOutputWrapper()) {
		LOGI("max pooling has output");
		inputWrapper = previousLayer->getOutputWrapper();
	} else {
		LOGI("max pooling doesn t have output");
		float *upstreamOutput = previousLayer->getOutput();
		inputWrapper = cl->wrap(previousLayer->getOutputNumElements(), upstreamOutput);
		inputWrapper->copyToDevice();
	}
	poolingForwardImpl->forward(batchSize, inputWrapper, selectorsWrapper, outputWrapper);

#endif

#if TEST_FORWARD_PROP == 0
	outputWrapper= previousLayer->getOutputWrapper();
	selectorsWrapper= previousLayer->getSelectorWrapper();
#endif
//
//if (1){
//outputWrapper= previousLayer->getOutputWrapper();
////if (typeid(previousLayer) == typeid(ActivationLayer))
//	selectorsWrapper= previousLayer->getSelectorWrapper();
//}else{
////25/12
//
//    //CLWrapper *upstreamOutputWrapper = 0;
//    if(previousLayer->hasOutputWrapper()) {
//    	LOGI("max pooling has output");
//    	inputWrapper = previousLayer->getOutputWrapper();
//    } else {
//    	LOGI("max pooling doesn t have output");
//        float *upstreamOutput = previousLayer->getOutput();
//        inputWrapper = cl->wrap(previousLayer->getOutputNumElements(), upstreamOutput);
//        inputWrapper->copyToDevice();
//    }
////    inputWrapper->copyToHost();
////    	float * grad0=(float*)inputWrapper->getHostArray();
////    		//inputWrapper->copyToHost();
////    		LOGI("////////////inputWrapper////////");
////    		for (int i =0;i<20/*dim.outputSize/2*/;i++){
////    			string displayArraY="";
////    			for (int j =0;j<14;j++){
////    				displayArraY= displayArraY+ "-" + to_string(grad0[i*14+j]);
////    			}
////    			LOGI("%s",displayArraY.c_str());
////    			displayArraY.clear();
////    		}
//    poolingForwardImpl->forward(batchSize, inputWrapper, selectorsWrapper, outputWrapper);
////    if(!previousLayer->hasOutputWrapper()) {
////        delete upstreamOutputWrapper;
////    }
//}

}
VIRTUAL void PoolingLayer::backward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: backward");
#endif

    // have no weights to backprop to, just need to backprop the errors

    CLWrapper *gradOutputWrapper = 0;
    bool weOwnErrorsWrapper = false;
    if(nextLayer->providesGradInputWrapper()) {
		#if TEST_BACKWARD_PROP == 1
			LOGI("on GPU");
		#endif
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
		#if TEST_BACKWARD_PROP == 1
			LOGI("on CPU");
		#endif
        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
        gradOutputWrapper->copyToDevice();
        weOwnErrorsWrapper = true;
    }

//	if (dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))
//		inputWrapper=((dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))->gradInputWrapper);
//
	#if TEST_FORWARD_PROP == 0
		if (dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))
			inputWrapper=(dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))->gradInput_poolingLayer_Wrapper;
		else
			if (dynamic_cast<ConvolutionalLayer*>(previousLayer))
				inputWrapper=(dynamic_cast<ConvolutionalLayer*>(previousLayer))->gradInput_poolingLayer_Wrapper;
			//inputWrapper=((dynamic_cast<ConvolutionalLayer*>(previousLayer->previousLayer))->gradInputWrapper);
	#endif
    poolingBackpropImpl->backward(batchSize, gradOutputWrapper, selectorsWrapper, gradInputWrapper, inputWrapper);



    if(weOwnErrorsWrapper) {
        delete gradOutputWrapper;
    }
}
VIRTUAL std::string PoolingLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/pooling/PoolingLayer.cpp: string PoolingLayer::asString");
#endif


    return "PoolingLayer{ inputPlanes=" + toString(numPlanes) + " inputSize=" + toString(inputSize) + " poolingSize=" + toString(poolingSize) + " }";
}


