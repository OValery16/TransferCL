// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>

#include "../../EasyCL/EasyCL.h"
#include "../net/NeuralNet.h"
#include "../util/stringhelper.h"
#include "Trainer.h"
#include "../batch/NetAction.h"
#include "TrainerStateMaker.h"
#include "TrainerState.h"
#include "../layer/Layer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL


Trainer::Trainer(EasyCL *cl) :
    cl(cl),
    learningRate(0) {
}
VIRTUAL Trainer::~Trainer() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: ~Trainer");
#endif


}
VIRTUAL void Trainer::setLearningRate(float learningRate) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: setLearningRate");
#endif


    this->learningRate = learningRate;
}
VIRTUAL std::string Trainer::asString() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: string Trainer::asString");
#endif


    return "Trainer{ learningRate=" + toString(learningRate) + " }";
}
VIRTUAL BatchResult Trainer::train(Trainable *trainable, 
        TrainingContext *context,
        float const*input, float const*expectedOutput) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: train");
#endif

    float loss = 0;

        NeuralNet *net = dynamic_cast< NeuralNet * > (trainable);
        return this->trainNet(net, context, input, expectedOutput);
    return BatchResult(loss, 0);
}
VIRTUAL BatchResult Trainer::trainFromLabels(Trainable *trainable,
    TrainingContext *context,
    float const*input, int const*labels) {

	#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/trainers/Trainer.cpp: trainFromLabels");
	#endif

    float loss = 0;
    int numRight = 0;
	NeuralNet *net = dynamic_cast< NeuralNet * > (trainable);
	return this->trainNetFromLabels(net, context, input, labels);
    return BatchResult(loss, numRight);
}
VIRTUAL void Trainer::_bindState(NeuralNet *net, TrainerStateMaker *stateMaker) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/trainers/Trainer.cpp: _bindState");
#endif

    // go through network layers, and assign TrainerState objects
#if TRANSFER==1
    for(int layerIdx = net->getNumLayers()-2; layerIdx < net->getNumLayers(); layerIdx++) {
#endif
#if TRANSFER==0
    for(int layerIdx = 0; layerIdx < net->getNumLayers(); layerIdx++) {
#endif
        Layer *layer = net->getLayer(layerIdx);
        if(layer->needsTrainerState()) {
            TrainerState *state = layer->getTrainerState();
            if(!stateMaker->created(state) ) {
                layer->setTrainerState(stateMaker);
            }
        }
    }
}

