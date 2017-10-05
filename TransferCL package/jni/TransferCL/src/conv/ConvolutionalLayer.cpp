// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "ConvolutionalLayer.h"
#include "ConvolutionalMaker.h"
#include "../net/NeuralNet.h"
#include "../util/stringhelper.h"
//#include "Forward.h"
#include "../weights/WeightsHelper.h"
//#include "BackpropWeights.h"
#include "../trainers/TrainerStateMaker.h"
#include "../trainers/TrainerState.h"
#include "../trainers/SGDState.h"
#include "../layer/Layer.h"
#include <string>

using namespace std;

#define VALIDATION_TEST_FORWARD_PROP 0 // if equal to 1, perform test but you need to activate TEST_FORWARD in Forward1.h
#define MEASURE_BACKWARD_PROP 0

#undef VIRTUAL
#define VIRTUAL

std::string to_string_with_precision7(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}

ConvolutionalLayer::ConvolutionalLayer(EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker) :
        Layer(previousLayer, maker),
        cl(cl),
        trainerState(0),
        biasTrainerState(0),
        forwardImpl(0),
        backwardImpl(0),
        inputWrapper(0),
        weights(0),
        bias(0),
        output(0),
        gradInput(0),
        gradWeights(0),
        gradBias(0),
        setup(false),
        weightsWrapper(0),
        biasWrapper(0),
        outputWrapper(0),
        selectorWrapper(0),
        gradInputWrapper(0),
        gradWeightsWrapper(0),
        gradBiasWrapper(0),
        batchSize(maker->_batchSize),
        allocatedSpaceNumExamples(0),
        isFC(false),
        setupBackprop(false),

		temp4(0),
		testWrapper(0)
            {
	//LOGI("conv batchSize %d",batchSize);
    dim.setNeedToNormalize(previousLayer->isFirstLayer())
    	.setIsConv(true)
    	.setBatchsize(batchSize)
    	.setActivationLayer(maker->_activationLayer)
    	.setTranslate(previousLayer->getTranslate())
    	.setScale(previousLayer->getScale())
    	.setStride(maker->_stride)
    	.setInputPlanes(previousLayer->getOutputPlanes())
        .setInputSize(previousLayer->getOutputSize())
        .setNumFilters(maker->_numFilters)
        .setFilterSize(maker->_filterSize)
        .setBiased(maker->_biased)
        .setPadZeros(maker->_padZeros)
        .setUseMaxPooling(maker->_useMaxPooling)
        .setMaxPool_spatialExtent(maker->_maxPool_spatialExtent)
        .setMaxPool_strides(maker->_maxPool_strides);

    isLast=maker->_isLast;

	#if TEST_UPDATE==1
		temp4= new float[dim.filtersSize];
		testWrapper=cl->wrap(dim.filtersSize, temp4);
		testWrapper->createOnDevice();

		temp5= new float[dim.numFilters];
		biasTestWrapper=cl->wrap(dim.numFilters, temp5);
		biasTestWrapper->createOnDevice();
	#endif


    if (dynamic_cast<ActivationLayer*>(previousLayer))
    	dim.setPreviousLayer_activationLayer((dynamic_cast<ActivationLayer*>(previousLayer))->fn->getDefineName());


    if (dim.useMaxPooling){

    	int * test=0;//we don t allocate anything in memory, the allocation test is delegate to OpenCL (wero-copy buffer)
    	selectorWrapper= cl->wrap(batchSize * dim.numFilters * dim.maxPool_sizeOutput*dim.maxPool_sizeOutput, test);
    	selectorWrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();


    }

    if(dim.padZeros && dim.filterSize % 2 == 0) {
    	LOGE("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize");
        throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
    }

    forwardImpl = new Forward1(previousLayer->isFirstLayer(),batchSize,cl, dim);
	#if TRANSFER==0
		backpropWeightsImpl = new BackpropWeightsNaive(cl, dim);
		if(previousLayer->needsBackProp()) {
			backwardImpl = new BackwardGpuNaive(cl, dim);
		}
	#endif


	weightsWrapper= cl->wrap(getWeightsSize() , weights);
	weightsWrapper->createZeroCopyObject_ReadWriteFlag_OnDevice();
	if(dim.biased) {
		biasWrapper= cl->wrap(getBiasSize() , bias);
		biasWrapper->createZeroCopyObject_ReadWriteFlag_OnDevice();
		bias=biasWrapper->map_ZeroCopyObject_WriteFlag();
	}
	weights=weightsWrapper->map_ZeroCopyObject_WriteFlag();
	randomizeWeights(maker->_weightsInitializer);

	weightsWrapper->unMap_ZeroCopyObject_WriteFlag(weights);
	if(dim.biased) {
		biasWrapper->unMap_ZeroCopyObject_WriteFlag(bias);
	}

#if MEASURE_BACKWARD_PROP==1
    gradWeights = new float[ getWeightsSize() ];
    gradWeightsWrapper = cl->wrap(getWeightsSize(), gradWeights);
    gradWeightsWrapper->createOnDevice();

    if(dim.biased) {
        gradBias = new float[ getBiasSize() ];
        gradBiasWrapper = cl->wrap(getBiasSize(), gradBias);
        gradBiasWrapper->createOnDevice();
    }

#endif

}
ConvolutionalLayer::ConvolutionalLayer(EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker,string s) :
        Layer(previousLayer, maker),
        cl(cl),
        trainerState(0),
        biasTrainerState(0),
        forwardImpl(0),
        backwardImpl(0),
        setup(false),
        weights(0),
        bias(0),
        output(0),
        gradInput(0),
        gradWeights(0),
        selectorWrapper(0),
        gradBias(0),
        weightsWrapper(0),
        biasWrapper(0),
        outputWrapper(0),
        gradInputWrapper(0),
        gradWeightsWrapper(0),
        gradBiasWrapper(0),
        batchSize(maker->_batchSize),
        allocatedSpaceNumExamples(0),
        isFC(true),
        setupBackprop(false),

		temp4(0),
		testWrapper(0)
            {
		//LOGI("conv batchSize %d",batchSize);
		//LOGI("fc 1");
		dim.setIsConv(false)
    	.setBatchsize(batchSize)
    	.setActivationLayer(maker->_activationLayer)
    	.setStride(maker->_stride)
    	.setInputPlanes(previousLayer->getOutputPlanes())
        .setInputSize(previousLayer->getOutputSize())
        .setNumFilters(maker->_numFilters)
        .setFilterSize(maker->_filterSize)
        .setBiased(maker->_biased)
        .setPadZeros(maker->_padZeros)
        .setUseMaxPooling(maker->_useMaxPooling)
        .setMaxPool_spatialExtent(maker->_maxPool_spatialExtent)
        .setMaxPool_strides(maker->_maxPool_strides);

	#if TEST_UPDATE==1
		temp4= new float[dim.filtersSize];
		testWrapper=cl->wrap(dim.filtersSize, temp4);
		testWrapper->createOnDevice();

		temp5= new float[dim.numFilters];
		biasTestWrapper=cl->wrap(dim.numFilters, temp5);
		biasTestWrapper->createOnDevice();
	#endif
		isLast=maker->_isLast;



    if (dynamic_cast<ActivationLayer*>(previousLayer))
    	dim.setPreviousLayer_activationLayer((dynamic_cast<ActivationLayer*>(previousLayer))->fn->getDefineName());


    if(dim.padZeros && dim.filterSize % 2 == 0) {
        throw std::runtime_error("filter size must be an odd number, if padZeros is true, so either turn off padZeros, or choose a different filtersize :-)");
    }
    forwardImpl = new Forward1(previousLayer->isFirstLayer(),batchSize,cl, dim);
	#if TRANSFER==0
		backpropWeightsImpl = new BackpropWeightsNaive(cl, dim);
		if(previousLayer->needsBackProp()) {
			backwardImpl = new BackwardGpuNaive(cl, dim);
		}
	#endif
    #if TRANSFER==1
        if (isLast){
			backpropWeightsImpl = new BackpropWeightsNaive(cl, dim);
			if(previousLayer->needsBackProp()) {
				backwardImpl = new BackwardGpuNaive(cl, dim);
			}
        }
	#endif
    if(dim.filterSize > dim.inputSize) {
            throw std::runtime_error("filter size cannot be larger than upstream image size: " + toString(dim.filterSize) +
                " > " + toString(dim.inputSize));
    }

	weightsWrapper= cl->wrap(getWeightsSize() , weights);
	weightsWrapper->createZeroCopyObject_ReadWriteFlag_OnDevice();
	if(dim.biased) {
		biasWrapper= cl->wrap(getBiasSize() , bias);
		biasWrapper->createZeroCopyObject_ReadWriteFlag_OnDevice();
		bias=biasWrapper->map_ZeroCopyObject_WriteFlag();
	}
	weights=weightsWrapper->map_ZeroCopyObject_WriteFlag();
	randomizeWeights(maker->_weightsInitializer);

	weightsWrapper->unMap_ZeroCopyObject_WriteFlag(weights);
	if(dim.biased) {
		biasWrapper->unMap_ZeroCopyObject_WriteFlag(bias);
	}


#if MEASURE_BACKWARD_PROP==1
    gradWeights = new float[ getWeightsSize() ];
    gradWeightsWrapper = cl->wrap(getWeightsSize(), gradWeights);
    gradWeightsWrapper->createOnDevice();

    if(dim.biased) {
        gradBias = new float[ getBiasSize() ];
        gradBiasWrapper = cl->wrap(getBiasSize(), gradBias);
        gradBiasWrapper->createOnDevice();
    }
#endif


}

VIRTUAL ConvolutionalLayer::~ConvolutionalLayer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: ~ConvolutionalLayer");
#endif
	#if TEST_UPDATE==1
		delete [] temp4;
		delete testWrapper;

		delete [] temp5;
		delete biasTestWrapper;
	#endif


    delete weightsWrapper;
    delete biasWrapper;
#if TRANSFER==0
    if (dim.useMaxPooling)
    	delete gradInput_poolingLayer_Wrapper;
    delete gradInputWrapper;
#endif
#if TRANSFER==1
    if(dim.isConv) {
		if (dim.useMaxPooling)
			delete gradInput_poolingLayer_Wrapper;
    }
	if (isLast)
		delete gradInputWrapper;

#endif

#if MEASURE_BACKWARD_PROP==1
    delete gradWeightsWrapper;
    delete gradBiasWrapper;
    delete[] gradWeights;
    delete[] gradBias;
#endif


#if TRANSFER==0
    delete backpropWeightsImpl;
    delete backwardImpl;
#endif
#if TRANSFER==1
    if (isLast)
		if(not dim.isConv) {
			delete[] gradInput;
			delete backpropWeightsImpl;
			delete backwardImpl;
		}
#endif



    delete outputWrapper;
    delete forwardImpl;
    delete trainerState;
    delete biasTrainerState;

    if (dim.useMaxPooling){
       delete selectorWrapper;
    }

}
VIRTUAL std::string ConvolutionalLayer::getClassName() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: string ConvolutionalLayer::getClassName");
#endif


    return "ConvolutionalLayer";
}

VIRTUAL float *ConvolutionalLayer::getGradInput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradInput");
#endif


    if(gradInputWrapper->isDeviceDirty()) {
        gradInputWrapper->copyToHost();
    }
    return gradInput;

}
VIRTUAL float *ConvolutionalLayer::getGradWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradWeights");
#endif

#if MEASURE_BACKWARD_PROP==1
    if(gradWeightsWrapper->isDeviceDirty()) {
        gradWeightsWrapper->copyToHost();
    }
    return gradWeights;

#endif
    return 0;
}
VIRTUAL float *ConvolutionalLayer::getGradBias() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradBias");
#endif

#if MEASURE_BACKWARD_PROP==1
    if(gradBiasWrapper->isDeviceDirty()) {
        gradBiasWrapper->copyToHost();
    }
    return gradBias;
#endif
return 0;
}
VIRTUAL bool ConvolutionalLayer::providesGradInputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: providesGradInputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradInputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradInputWrapper");
#endif


    return gradInputWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getWeightsWrapper");
#endif


    return weightsWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBiasWrapper");
#endif


    return biasWrapper;
}

CLWrapper *ConvolutionalLayer::getSelectorWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBiasWrapper");
#endif


    return selectorWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradWeightsWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradWeightsWrapper");
#endif


    return gradWeightsWrapper;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getGradBiasWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getGradBiasWrapper");
#endif


    return gradBiasWrapper;
}
VIRTUAL bool ConvolutionalLayer::hasOutputWrapper() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: hasOutputWrapper");
#endif


    return true;
}
VIRTUAL CLWrapper *ConvolutionalLayer::getOutputWrapper() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputWrapper");
#endif


    return outputWrapper;
}
VIRTUAL bool ConvolutionalLayer::needsBackProp() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: needsBackProp");
#endif


    return true;
}
VIRTUAL int ConvolutionalLayer::getOutputNumElements() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputNumElements");
#endif


    return batchSize * dim.outputCubeSize;
}
VIRTUAL int ConvolutionalLayer::getOutputPlanes() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputPlanes");
#endif


    return dim.numFilters;
}
VIRTUAL int ConvolutionalLayer::getOutputSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputSize");
#endif


    return dim.outputSize;
}

void ConvolutionalLayer::randomizeWeights(WeightsInitializer *weightsInitializer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: randomizeWeights");
#endif

    int fanin = dim.inputPlanes * dim.filterSize * dim.filterSize;
    if(dim.biased) {
        fanin++;
    }
    const int numThisLayerWeights = getWeightsSize();
    weightsInitializer->initializeWeights(numThisLayerWeights, weights, fanin);
    if(dim.biased) {
        weightsInitializer->initializeWeights(dim.numFilters, bias, fanin);
    }
}
VIRTUAL void ConvolutionalLayer::print() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: print");
#endif


    std::cout << "ConvolutionalLayer " << dim << std::endl;
    printWeights();
    if(output != 0) {
        printOutput();
    }
}
VIRTUAL void ConvolutionalLayer::printWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: printWeights");
#endif


    std::cout << "  weights: " << std::endl;
    getWeights();
    for(int filter = 0; filter < std::min(5, dim.numFilters); filter++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.numFilters); filter++) {");
#endif


       std::cout << "    filter " << filter << std::endl;
       if(dim.biased) {
           std::cout << "       bias=" << bias[filter] << std::endl;
       }
       for(int plane = 0; plane < std::min(5, dim.inputPlanes); plane++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.inputPlanes); plane++) {");
#endif


           if(dim.inputPlanes > 1) std::cout << "    inplane " << plane << std::endl;
            for(int i = 0; i < std::min(5, dim.filterSize); i++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.filterSize); i++) {");
#endif


                std::cout << "      ";
                for(int j = 0; j < std::min(5, dim.filterSize); j++) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.filterSize); j++) {");
#endif


                   std::cout << getWeight(filter, plane, i, j) << " ";
                }
                if(dim.filterSize > 5) {
                   std::cout << " ...";
                }
                std::cout << std::endl;
            }
            if(dim.filterSize > 5) {
               std::cout << " ..." << std::endl;
            }
        }
        if(dim.inputPlanes > 5) std::cout << " ... other inplanes ... " << std::endl;
    }
    if(dim.numFilters > 5) std::cout << " ... other filters ... " << std::endl;
 }
VIRTUAL void ConvolutionalLayer::printOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: printOutput");
#endif


//    if(output == 0) {
//        return;
//    }
//    //    getOutput();
//    std::cout << "  outputs: " << std::endl;
//// output are organized like [imageid][filterid][row][col]
//    for(int n = 0; n < std::min(5, batchSize); n++) {
//#if TRANSFERCL_VERBOSE == 1
//LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, batchSize); n++) {");
//#endif
//
//
//        std::cout << "    n: " << n << std::endl;
//        for(int plane = 0; plane < std::min(5, dim.numFilters); plane++) {
//#if TRANSFERCL_VERBOSE == 1
//LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.numFilters); plane++) {");
//#endif
//
//
//            if(dim.numFilters > 1) std::cout << "      plane " << plane << std::endl;
//            if(dim.outputSize == 1) {
//                 std::cout << "        " << getOutput(n, plane, 0, 0) << std::endl;
//            } else {
//                for(int i = 0; i < std::min(5, dim.outputSize); i++) {
//#if TRANSFERCL_VERBOSE == 1
//LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.outputSize); i++) {");
//#endif
//
//
//                    std::cout << "      ";
//                    for(int j = 0; j < std::min(5, dim.outputSize); j++) {
//#if TRANSFERCL_VERBOSE == 1
//LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: min(5, dim.outputSize); j++) {");
//#endif
//
//
//                        std::cout << getOutput(n, plane, i, j) << " ";
//                    }
//                    if(dim.outputSize > 5) std::cout << " ... ";
//                    std::cout << std::endl;
//                }
//                if(dim.outputSize > 5) std::cout << " ... " << std::endl;
//            }
//            if(dim.numFilters > 5) std::cout << " ... other planes ... " << std::endl;
//        }
//        if(batchSize > 5) std::cout << " ... other n ... " << std::endl;
//    }
}
VIRTUAL void ConvolutionalLayer::setBatchSize(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: setBatchSize");
#endif


    if(batchSize <= allocatedSpaceNumExamples) {
        this->batchSize = batchSize;
        return;
    }

    this->batchSize = batchSize;
    this->allocatedSpaceNumExamples = batchSize;


    if (not setup){


#if TRANSFER==0
    	float * test=0;//we don t allocate anything in memory, the allocation test is delegate to OpenCL (wero-copy buffer)
    	outputWrapper= cl->wrap(getOutputNumElements(), test);
    	outputWrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
    	if (dim.useMaxPooling){
    		gradInput_poolingLayer_Wrapper= cl->wrap(getOutputNumElements(), test);
    		gradInput_poolingLayer_Wrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
    	}
    	if(layerIndex > 1) {
    		gradInputWrapper= cl->wrap( previousLayer->getOutputNumElements(), test);
    		gradInputWrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
        }
#endif
#if TRANSFER==1
	    float * test=0;//we don t allocate anything in memory, the allocation test is delegate to OpenCL (wero-copy buffer)
    	outputWrapper= cl->wrap(getOutputNumElements(), test);
    	if (isLast)
    		outputWrapper->createZeroCopyObject_ReadFlag_OnDevice();
    	else
    		outputWrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
    	if (dim.useMaxPooling){
    		gradInput_poolingLayer_Wrapper= cl->wrap(getOutputNumElements(), test);
    		gradInput_poolingLayer_Wrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
    	}
    	if (isLast)
			if(not dim.isConv) {
				gradInputWrapper= cl->wrap( previousLayer->getOutputNumElements(), test);
				gradInputWrapper->createZeroCopyObject_HostNotAccessFlag_OnDevice();
			}
#endif

		setup=true;
		}

}
VIRTUAL void ConvolutionalLayer::setWeights(float *weights, float *bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: setWeights");
#endif


//    cout << "setweights" << endl;
    initWeights(weights);
    if(dim.biased) {
        initBias(bias);
    }
}
VIRTUAL int ConvolutionalLayer::getOutputCubeSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutputCubeSize");
#endif


    return dim.outputCubeSize;
}
VIRTUAL int ConvolutionalLayer::getPersistSize(int version) const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getPersistSize");
#endif


    if(dim.biased) {
        return getWeightsSize() + getBiasSize();
    } else {
        return getWeightsSize();
    }
}
VIRTUAL void ConvolutionalLayer::persistToArray(int version, float *array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: persistToArray");
#endif

#if DISPLAY_WEIGHT == 1
 	LOGI("######################################");
	LOGI("######## layer s weights #############");
	LOGI("######################################");

		string s="";
		for(int i1=0; i1<20;i1++){
			s=s+" "+ to_string(weights[i1]);
		}
		LOGI("%s",s.c_str());
#endif


	weights=weightsWrapper->map_ZeroCopyObject_ReadFlag();
    memcpy(array, weights, sizeof(float) * getWeightsSize());
    weightsWrapper->unMap_ZeroCopyObject_ReadFlag(weights);
    if(dim.biased) {
    	bias=biasWrapper->map_ZeroCopyObject_WriteFlag();
        memcpy(array + getWeightsSize(), bias, sizeof(float) * getBiasSize());
        biasWrapper->unMap_ZeroCopyObject_ReadFlag(weights);
		#if DISPLAY_WEIGHT == 1
				s="";
				for(int i1=0; i1<getBiasSize();i1++){
					s=s+" "+ to_string(bias[i1]);
				}
				LOGI("bias %s",s.c_str());
				LOGI("---------------------------->bias size %d",getBiasSize());
		#endif
	}

}
VIRTUAL void ConvolutionalLayer::unpersistFromArray(int version, float const*array) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: unpersistFromArray");
#endif

cl->finish();
    float const*newweights = array;

	#if DISPLAY_WEIGHT == 1
		LOGI("######################################");
		LOGI("######## CONV s weights #############");
		LOGI("######################################");
	#endif

    weights=weightsWrapper->map_ZeroCopyObject_WriteFlag();
    initWeights(newweights);
    weightsWrapper->unMap_ZeroCopyObject_WriteFlag(weights);
	#if DISPLAY_WEIGHT == 1
		string s="";
		for(int i1=0; i1<20;i1++){
			s=s+" "+ to_string(weights[i1]);
		}
		LOGI("%s",s.c_str());
	#endif
    if(dim.biased) {
        float const*newbias = array + getWeightsSize();
        bias=biasWrapper->map_ZeroCopyObject_WriteFlag();
        initBias(newbias);
        biasWrapper->unMap_ZeroCopyObject_WriteFlag(bias);
		#if DISPLAY_WEIGHT == 1
			s="";
			for(int i1=0; i1<getBiasSize();i1++){
				s=s+" "+ to_string(bias[i1]);
			}
			LOGI("bias %s",s.c_str());
			LOGI("---------------------------->bias size %d",dim.numFilters);
		#endif
    }
}
VIRTUAL void ConvolutionalLayer::initWeights(float const*weights) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: initWeights");
#endif

    int weightsSize = getWeightsSize();
    memcpy(this->weights, weights, sizeof(float) * weightsSize);

}
VIRTUAL void ConvolutionalLayer::initBias(float const*bias) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: initBias");
#endif


    int biasSize = dim.numFilters;
    memcpy(this->bias, bias, sizeof(float) * biasSize);

}
VIRTUAL int ConvolutionalLayer::getWeightsSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getWeightsSize");
#endif


    return dim.numFilters * dim.inputPlanes * dim.filterSize * dim.filterSize;
}
VIRTUAL int ConvolutionalLayer::getBiasSize() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBiasSize");
#endif


    if(dim.biased) {
        return dim.numFilters;
    } else {
        return 0;
    }
}
VIRTUAL float const *ConvolutionalLayer::getWeights() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getWeights");
#endif


    if(weightsWrapper->isDeviceDirty()) {
        throw std::runtime_error("weights not copied to host, and htis is const object, so cannot copy");
    }
    return weights;
}
VIRTUAL float *ConvolutionalLayer::getWeights() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getWeights");
#endif



	cl->finish();
	weightsWrapper->copyToHost();

    return weights;
}
VIRTUAL float *ConvolutionalLayer::getBias() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBias");
#endif


	cl->finish();
	biasWrapper->copyToHost();

    return bias;
}
VIRTUAL float const*ConvolutionalLayer::getBias() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBias");
#endif


    if(biasWrapper->isDeviceDirty()) {
        throw std::runtime_error("bias not copied to host, and htis is const object, so cannot copy");
    }
    return bias;
}

VIRTUAL float * ConvolutionalLayer::getOutput() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getOutput");
#endif


    if(outputWrapper->isDeviceDirty()) {
        outputWrapper->copyToHost();
		#if TRANSFERCL_VERBOSE == 1
        	LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: the output is in the gpu, we need to copy it back to the cpu");
		#endif
    }
    return output;
};

int ConvolutionalLayer::getInputIndexTest(int n, int plane, int row, int col) {
    return (( n
        * dim.numFilters + plane)
        * dim.outputSize + row)
        * dim.outputSize + col;
}
int ConvolutionalLayer::getResultIndexTest(int n, int plane, int row, int col) {
    return (( n
        * dim.numFilters + plane)
        * dim.maxPool_sizeOutput + row)
        * dim.maxPool_sizeOutput + col;
}

void ConvolutionalLayer::pollingtest(int batchSize, float *input, int *selectors, float *output) {

    for(int n = 0; n < batchSize; n++) {
        for(int plane = 0; plane < dim.numFilters; plane++) {
            for(int outputRow = 0; outputRow < ( dim.outputSize / dim.maxPool_strides); outputRow++) {
                int inputRow = outputRow * dim.maxPool_strides;
                for(int outputCol = 0; outputCol < ( dim.outputSize / dim.maxPool_strides); outputCol++) {
                    int inputCol = outputCol * dim.maxPool_strides;
                    int selector = 0;
                    float maxValue = input[ getInputIndexTest(n, plane, inputRow, inputCol) ];
                    for(int dx = 0; dx < dim.maxPool_strides; dx++) {
                        for(int dy = 0; dy < dim.maxPool_strides; dy++) {
                            if((inputRow + dx < dim.outputSize) && (inputCol + dy < dim.outputSize)) {
                                float thisValue = input[ getInputIndexTest(n, plane, inputRow + dx, inputCol + dy) ];
                                if(thisValue > maxValue) {
                                    maxValue = thisValue;
                                    selector = dx * dim.maxPool_strides + dy;
                                }
                            }
                        }
                    }
                    int resultIndex = getResultIndexTest(n, plane, outputRow, outputCol);
                    output[ resultIndex ] = maxValue;
                    selectors[ resultIndex ] = selector;
                }
            }
        }
    }

}
VIRTUAL void ConvolutionalLayer::forward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: forward");
#endif


    if(previousLayer->hasOutputWrapper()) {
    	inputWrapper = previousLayer->getOutputWrapper();
    } else {
    	inputWrapper = cl->wrap(previousLayer->getOutputNumElements(), (float *)previousLayer->getOutput());
    	inputWrapper->copyToDevice();

    }

	#if VALIDATION_TEST_FORWARD_PROP
		if(dim.test) {
			if (dim.useMaxPooling){

				float * temp2= new float[getOutputNumElements()];
				CLWrapper * temp2Wrapper = cl->wrap(getOutputNumElements(), (float *)temp2);
				temp2Wrapper->copyToDevice();

				float * temp = new float[batchSize * dim.outputCubeSize];
				forwardImpl->forwardHalf(batchSize, inputWrapper, weightsWrapper, biasWrapper, outputWrapper ,selectorWrapper,temp2Wrapper);

				temp2Wrapper->copyToHost();
				float * conv0=(float*)temp2Wrapper->getHostArray();
				outputWrapper->copyToHost();
				float * conv=(float*)outputWrapper->getHostArray();
				int col0=dim.outputSize;

				selectorWrapper->copyToHost();
				int * selector0=(int*)selectorWrapper->getHostArray();

				for (int i=0; i<batchSize * dim.outputCubeSize;i++)
					temp[i]=conv[i];

				float * input2=0;
				CLWrapper *upstreamWrapper2 = 0;
				if (previousLayer->isFirstLayer()){

					input2= new float[previousLayer->getOutputNumElements()];
					inputWrapper->copyToHost(input2);
					upstreamWrapper2 = cl->wrap(previousLayer->getOutputNumElements(), (float *)input2);
					for(int i = 0; i < previousLayer->getOutputNumElements(); i++) {
							input2[i] = (input2[i] + dim.translate) * dim.scale;
						}
					upstreamWrapper2->copyToDevice();




					//input2= new float[previousLayer->getOutputNumElements()];

					forwardImpl->forwardFloat(batchSize, upstreamWrapper2/*inputWrapper*/, weightsWrapper, biasWrapper, outputWrapper);
				}else
					forwardImpl->forwardFloat(batchSize, inputWrapper, weightsWrapper, biasWrapper, outputWrapper);
				outputWrapper->copyToHost();
				conv=(float*)outputWrapper->getHostArray();
				float sum=0.0f;

				int col=dim.outputSize;
		//



				for(int i =0;i<batchSize * dim.outputCubeSize; i++){
					if (conv[i]<0)
						conv[i]=0;

				}
				int *selectors = new int[ batchSize * dim.numFilters * ( dim.maxPool_sizeOutput)* ( dim.maxPool_sizeOutput) ];
				float *output = new float[batchSize * dim.numFilters * ( dim.maxPool_sizeOutput)* ( dim.maxPool_sizeOutput) ];
				pollingtest(batchSize, conv, selectors, output);

				int col2=dim.maxPool_sizeOutput;

				LOGI("////////////conv + pool output////////");
				for (int i =0;i<dim.outputSize;i++){
					string displayArraY="";
					for (int j =0;j<dim.outputSize;j++){
						displayArraY= displayArraY+ "-" + to_string_with_precision7(conv[i*col0+j]);
					}
					LOGI("%s",displayArraY.c_str());
					displayArraY.clear();
				}

						LOGI("////////////selectors////////");
						for (int i =0;i<20/*dim.outputSize/2*/;i++){
							string displayArraY="";
							for (int j =0;j<dim.maxPool_sizeOutput;j++){
								displayArraY= displayArraY+ "-" + to_string(selectors[i*col2+j]);
							}
							LOGI("%s",displayArraY.c_str());
							displayArraY.clear();
						}

				sum=0;
				for(int i =0;i<batchSize*dim.outputCubeSize; i++){
					sum+=abs(conv[i]-conv0[i]);
				}
				LOGE("cal 1 error diff %f",sum);
				sum=0;
							for(int i =0;i<dim.outputSizeSquared; i++){
								sum+=abs(conv[i]-conv0[i]);
							}
							LOGE("partial cal 1 error diff %f",sum);
				sum=0;
				for(int i =0;i<batchSize*dim.numFilters*dim.maxPool_sizeOutput*dim.maxPool_sizeOutput; i++){
					sum+=abs(temp[i]-output[i]);
				}
				LOGE("col2 %d diff %f",col2,sum);
				sum=0;
				for(int i =0;i<batchSize*dim.numFilters*dim.maxPool_sizeOutput*dim.maxPool_sizeOutput; i++){
					sum+=abs(selector0[i]-selectors[i]);

				}
				LOGE("selector diff %f",sum);
				delete[] temp;
				delete[] selectors;
				delete[] output;
				delete [] temp2;
				delete temp2Wrapper;
				if (previousLayer->isFirstLayer()){
					delete [] input2;
					delete upstreamWrapper2;
				}
			}else{
					float * temp = new float[batchSize * dim.outputCubeSize];

					forwardImpl->forwardHalf(batchSize, inputWrapper, weightsWrapper, biasWrapper, outputWrapper,selectorWrapper,gradInputWrapper);
					outputWrapper->copyToHost();
					float * conv=(float*)outputWrapper->getHostArray();

					for (int i=0; i<batchSize * dim.outputCubeSize;i++)
						temp[i]=conv[i];
					int col0=dim.outputSize;
							for (int i =0;i<20/*dim.outputSize/2*/;i++){
								string displayArraY="";
								for (int j =0;j<dim.outputSize/2;j++){
									displayArraY= displayArraY+ "-" + to_string_with_precision7(temp[i*col0+j]);
								}
								LOGI("%s",displayArraY.c_str());
								displayArraY.clear();
							}

					forwardImpl->forwardFloat(batchSize, inputWrapper, weightsWrapper, biasWrapper, outputWrapper);

					outputWrapper->copyToHost();
					conv=(float*)outputWrapper->getHostArray();
					float sum=0.0f;

					int col=dim.outputSize;
					LOGI("////////////conv////////");
					for (int i =0;i<20;i++){
						string displayArraY="";
						for (int j =0;j<dim.outputSize/2;j++){
							displayArraY= displayArraY+ "-" + to_string_with_precision7(conv[i*col+j]);
						}
						LOGI("%s",displayArraY.c_str());
						displayArraY.clear();
					}
					LOGI("////////////conv////////");
					for(int i =0;i<batchSize * dim.outputCubeSize; i++){

						sum+=abs(temp[i]-tanh(conv[i]));
					}
					int *selectors = new int[ batchSize * dim.numFilters * ( dim.outputSize / 2)* ( dim.outputSize / 2) ];
					float *output = new float[batchSize * dim.numFilters * ( dim.outputSize / 2)* ( dim.outputSize / 2) ];

					LOGE("diff %f",sum);
					delete[] temp;
					delete[] selectors;
					delete[] output;

			}
		}
	#endif
	forwardImpl->forwardHalf(batchSize, inputWrapper, weightsWrapper, biasWrapper, outputWrapper,selectorWrapper,gradInput_poolingLayer_Wrapper/*gradInputWrapper*/);



}
VIRTUAL void ConvolutionalLayer::backward() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: backward");
#endif

#if MEASURE_BACKWARD_PROP==1
	struct timeval start1, end1;
    gettimeofday(&start1, NULL);

#endif
	if (not setupBackprop){
		dim.setMomentum(momentum);
		dim.setLearningRate(learning_rate);
		dim.setWeightDecay(weightDecay);
		backpropWeightsImpl->dim.setMomentum(momentum);
		backpropWeightsImpl->dim.setLearningRate(learning_rate);
		backpropWeightsImpl->dim.setWeightDecay(weightDecay);
		setupBackprop=true;
	}


    CLWrapper *gradOutputWrapper = 0;
    bool weOwnGradOutputWrapper = false;
    if(nextLayer->providesGradInputWrapper()) {
        gradOutputWrapper = nextLayer->getGradInputWrapper();
    } else {
        gradOutputWrapper = cl->wrap(getOutputNumElements(), nextLayer->getGradInput());
        gradOutputWrapper->copyToDevice();
        weOwnGradOutputWrapper = true;
    }
#if MEASURE_BACKWARD_PROP==1
    cl->finish();
    gettimeofday(&end1, NULL);
    LOGI("--------------------------preloading took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));
#endif




    if(previousLayer->needsBackProp()) {
		#if TRANSFERCL_VERBOSE == 1
			LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: --- backward layer %s , needBackProp",toString(layerIndex).c_str());
		#endif

		#if MEASURE_BACKWARD_PROP==1
			gettimeofday(&start1, NULL);
		#endif


    	backwardImpl->backward(batchSize, inputWrapper, gradOutputWrapper, weightsWrapper, gradInputWrapper);
		#if MEASURE_BACKWARD_PROP==1
    		cl->finish();
    	    gettimeofday(&end1, NULL);
    	    LOGI("--------------------------backward computation took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));
		#endif
		#if TRANSFERCL_VERBOSE == 1
			LOGI("DeepCL/src/conv/ConvolutionalLayer.cpp: --- backward layer %s ,  after clFinish",toString(layerIndex).c_str());
		#endif
    }

	#if MEASURE_BACKWARD_PROP==1
    	gettimeofday(&start1, NULL);
	#endif
	CLWrapper *previousStepVectorWrapper=dynamic_cast< SGDState * >(getTrainerState())->lastUpdateWrapper;
	CLWrapper *previousStepBiasVectorWrapper=dynamic_cast< SGDState * >(getBiasTrainerState())->lastUpdateWrapper;

	#if TEST_UPDATE==1
		weightsWrapper->copyTo(testWrapper);
		biasWrapper->copyTo(biasTestWrapper);
		backpropWeightsImpl->calcGradWeights(batchSize, gradOutputWrapper, inputWrapper,  gradWeightsWrapper, gradBiasWrapper, testWrapper/*weightsWrapper*/, previousStepVectorWrapper , biasTestWrapper/*biasWrapper*/,previousStepBiasVectorWrapper);
	#endif
	#if TEST_UPDATE==0
    	backpropWeightsImpl->calcGradWeights(batchSize, gradOutputWrapper, inputWrapper,  gradWeightsWrapper, gradBiasWrapper, weightsWrapper, previousStepVectorWrapper , biasWrapper,previousStepBiasVectorWrapper);
	#endif

    #if MEASURE_BACKWARD_PROP==1
    	cl->finish();
    	gettimeofday(&end1, NULL);
    	LOGI("--------------------------propagation to weights computation took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));
    	gettimeofday(&start1, NULL);
	#endif
    if(!previousLayer->hasOutputWrapper()) {
        delete inputWrapper;
    }
    if(weOwnGradOutputWrapper) {
        delete gradOutputWrapper;
    }
	#if MEASURE_BACKWARD_PROP==1
    cl->finish();
        gettimeofday(&end1, NULL);
        LOGI("--------------------------cleaning computation took %f\n ms", (float)(((end1.tv_sec * 1000000 + end1.tv_usec)	- (start1.tv_sec * 1000000 + start1.tv_usec))/1000));
	#endif

}

VIRTUAL std::string ConvolutionalLayer::asString() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: string ConvolutionalLayer::asString");
#endif


    return "ConvolutionalLayer{ " + toString(dim) + " }";
}
VIRTUAL bool ConvolutionalLayer::needsTrainerState() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: needsTrainerState");
#endif


    return true;
}
VIRTUAL bool ConvolutionalLayer::biased() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: biased");
#endif


    return dim.biased;
}
VIRTUAL TrainerState *ConvolutionalLayer::getTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getTrainerState");
#endif


    return trainerState;
}
VIRTUAL TrainerState *ConvolutionalLayer::getBiasTrainerState() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: getBiasTrainerState");
#endif


    return biasTrainerState;
}
VIRTUAL void ConvolutionalLayer::setTrainerState(TrainerStateMaker *trainerStateMaker) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/ConvolutionalLayer.cpp: setTrainerState");
#endif


    delete trainerState;
    delete biasTrainerState;
    this->trainerState = trainerStateMaker->instance(cl, getWeightsSize());
    if(dim.biased) {
        this->biasTrainerState = trainerStateMaker->instance(cl, getBiasSize());
    }
}

bool ConvolutionalLayer::isConvLayer() const {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/Layer.cpp: isConvLayer");
#endif


    return not isFC;
}
