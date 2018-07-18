// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com // Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include "../dependencies.h"
#include "../layer/Layer.h"
#include "../../EasyCL/EasyCL.h"
//#include "ClConvolve2.h"
#include "../activate/ActivationLayer.h"
#include "../activate/ActivationFunction.h"
#include "../layer/LayerMaker.h"
#include "../util/Timer.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/stringhelper.h"
#include "LayerDimensions.h"
#include "../trainers/TrainerState.h"
#include <math.h>
#include "Forward1.h"
#include "BackwardGpuNaive.h"
#include "BackpropWeightsNaive.h"

#define VIRTUAL virtual

class TrainerStateMaker;
class Forward;
class Backward;
class BackpropWeights;
class ConvolutionalMaker;
class GpuAdd;
class CopyBuffer;
class WeightsInitializer;

class ConvolutionalLayer : public Layer {
public:
    EasyCL *const cl; // NOT owned by us
    TrainerState *trainerState; // OWNED by us, we should delete (if non-zero)
    TrainerState *biasTrainerState; // OWNED by us, we should delete (if non-zero)

    Forward1 *forwardImpl;
    BackpropWeightsNaive *backpropWeightsImpl;
    BackwardGpuNaive *backwardImpl;

    LayerDimensions dim;
//    ActivationFunction const *const activationFunction;

    float *weights;
    float *bias;
    float *output;
    float *gradInput;
    float *gradWeights;
    float *gradBias;
    bool isFC;
    bool isLast;

    int * selector;

    CLWrapper *testWrapper;
    float *temp4;
    CLWrapper *biasTestWrapper;
    float *temp5;

    CLWrapper *selectorWrapper;
    CLWrapper *weightsWrapper;
    CLWrapper *biasWrapper;
    CLWrapper *outputWrapper;
    CLWrapper *gradInputWrapper;
    CLWrapper *gradWeightsWrapper;
    CLWrapper *gradBiasWrapper;
    CLWrapper *inputWrapper;

    CLWrapper *gradInput_poolingLayer_Wrapper;

    int batchSize;
    int allocatedSpaceNumExamples;

    bool setup;
    bool setupBackprop;


    inline int getWeightIndex(int filterId, int inputPlane, int filterRow, int filterCol) const {
        return (( filterId 
            * dim.inputPlanes + inputPlane)
            * dim.filterSize + filterRow)
            * dim.filterSize + filterCol;
    }
    inline float getWeight(int filterId, int inputPlane, int filterRow, int filterCol) const {
//        getWeights();
        return weights[ getWeightIndex(filterId, inputPlane, filterRow, filterCol) ];
    }
    inline int getOutputIndex(int n, int outPlane, int outRow, int outCol) const {
        return (( n
            * dim.numFilters + outPlane)
            * dim.outputSize + outRow)
            * dim.outputSize + outCol;
    }
    inline float getOutput(int n, int outPlane, int outRow, int outCol) const {
        return output[ getOutputIndex(n,outPlane, outRow, outCol) ];
    }

//    ConvolutionalLayer(Layer *previousLayer, ConvolutionalMaker const*maker);
    // images are organized like [imageId][plane][imagerow][imagecol]
    // filters are organized like [filterid][plane][filterrow][filtercol]
    // output are organized like [imageid][filterid][imagerow][imagecol]
//    inline int getWeightIndex(int outPlane, int inPlane, int filterrow, int filtercol) const {
//        return (( outPlane * upstreamNumPlanes 
//             + inPlane) * filterSize 
//             + filterrow) * filterSize
//             + filtercol;
//    }
//    inline float getWeight(int outPlane, int inPlane, int filterrow, int filtercol) const {
//        return weights[getWeightIndex(outPlane, inPlane, filterrow, filtercol) ];
//    }

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    ConvolutionalLayer(EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker);
    ConvolutionalLayer(EasyCL *cl, Layer *previousLayer, ConvolutionalMaker *maker,string s);
    int getActivationFunctionIdx();
    VIRTUAL ~ConvolutionalLayer();
    VIRTUAL std::string getClassName() const;
    VIRTUAL float *getGradInput();
    VIRTUAL float *getGradWeights();
    VIRTUAL float *getGradBias();
    VIRTUAL bool providesGradInputWrapper() const;
    VIRTUAL CLWrapper *getGradInputWrapper();
    VIRTUAL CLWrapper *getWeightsWrapper();
    VIRTUAL CLWrapper *getBiasWrapper();
    VIRTUAL CLWrapper *getGradWeightsWrapper();
    VIRTUAL CLWrapper *getGradBiasWrapper();
    VIRTUAL bool hasOutputWrapper() const;
    VIRTUAL CLWrapper *getOutputWrapper();
    CLWrapper *getSelectorWrapper();
    VIRTUAL bool needsBackProp();
    VIRTUAL int getOutputNumElements() const;
    VIRTUAL int getOutputPlanes() const;
    VIRTUAL int getOutputSize() const;
    void randomizeWeights(WeightsInitializer *weightsInitializer);
    VIRTUAL void print();
    VIRTUAL void printWeights();
    VIRTUAL void printOutput();
    VIRTUAL void setBatchSize(int batchSize);
    VIRTUAL void setWeights(float *weights, float *bias);
    VIRTUAL int getOutputCubeSize() const;
    VIRTUAL int getPersistSize(int version) const;
    VIRTUAL void persistToArray(int version, float *array);
    VIRTUAL void unpersistFromArray(int version, float const*array);
    VIRTUAL void initWeights(float const*weights);
    VIRTUAL void initBias(float const*bias);
    VIRTUAL int getWeightsSize() const;
    VIRTUAL int getBiasSize() const;
    VIRTUAL float const *getWeights() const;
    VIRTUAL float *getWeights();
    VIRTUAL float *getBias();
    VIRTUAL float const*getBias() const;
    VIRTUAL float * getOutput();
    VIRTUAL void forward();
    VIRTUAL void backward();
    VIRTUAL std::string asString() const;
    VIRTUAL bool needsTrainerState() const;
    VIRTUAL bool biased();
    VIRTUAL TrainerState *getTrainerState();
    VIRTUAL TrainerState *getBiasTrainerState();
    VIRTUAL void setTrainerState(TrainerStateMaker *trainerStateMaker);
    VIRTUAL bool isConvLayer() const;

    int getInputIndexTest(int n, int plane, int row, int col);
    int getResultIndexTest(int n, int plane, int row, int col);
    void pollingtest(int batchSize, float *input, int *selectors, float *output);
    // [[[end]]]
};

