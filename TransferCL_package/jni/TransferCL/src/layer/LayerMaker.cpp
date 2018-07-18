// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "../net/NeuralNet.h"
#include "../fc/FullyConnectedLayer.h"
#include "../conv/ConvolutionalLayer.h"
#include "../input/InputLayer.h"
#include "../loss/SoftMaxLayer.h"
#include "../loss/SoftMaxLayerPredict.h"
#include "../loss/SquareLossLayer.h"
#include "../loss/CrossEntropyLoss.h"
#include "../pooling/PoolingLayer.h"
#include "../normalize/NormalizationLayer.h"


#include "LayerMaker.h"

using namespace std;

Layer *SquareLossMaker::createLayer(Layer *previousLayer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/LayerMaker.cpp: createLayer");
#endif


    return new SquareLossLayer(previousLayer, this);
}
Layer *CrossEntropyLossMaker::createLayer(Layer *previousLayer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/LayerMaker.cpp: createLayer");
#endif


    return new CrossEntropyLoss(previousLayer, this);
}
Layer *SoftMaxMaker::createLayer(Layer *previousLayer) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/layer/LayerMaker.cpp: createLayer");
#endif
	if (this->prediction)
		return new SoftMaxLayerPredict(previousLayer, this,this->batchSize);
	else
		return new SoftMaxLayer(previousLayer, this,this->batchSize);
}

