// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../sonyOpenCLexample1.h"
#include <string>

//#include "../DeepCLDllExport.h"

class NeuralNet;
class WeightsInitializer;



/// \brief Add layers to a NeuralNet object, based on a netdef-string
///
/// eg "8c5-mp2" will add a convolutional layer with 8 filter, each 
/// 5 by 5; and one max-pooling layer, over 2x2
/// based on the notation proposed in 
/// [Multi-column Deep Neural Networks for Image Classification](http://arxiv.org/pdf/1202.2745.pdf)
class NetdefToCompile {
public:

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
	static bool createNetForCompile(int batchsize,NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer);
	static void compileConv(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,std::string pooling_layerDef, WeightsInitializer *weightsInitializer);
	static void compileMaxPooling(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
	static void compileDropout(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
	static void compileActivation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
	static void compileRandomTranslation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer);
	static void compileFullyConnectedLayer(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef, WeightsInitializer *weightsInitializer);


    // [[[end]]]
};

