// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <stdexcept>

#include "../layer/Layer.h"
#include "NeuralNet.h"
#include "../input/InputLayerMaker.h"
#include "../../EasyCL/EasyCL.h"

#include "NeuralNetMould.h"

using namespace std;

NeuralNet *NeuralNetMould::instance() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/net/NeuralNetMould.cpp: instance");
#endif


//    cout << "neuralnetmould::instance imagesize " << _imageSize << " numPlanes " << _numPlanes << endl;
    if(_numPlanes != 0 || _imageSize != 0) {
        if(_numPlanes == 0) {
            throw runtime_error("Must provide ->planes(planes)");
        }
        if(_imageSize == 0) {
            throw runtime_error("Must provide ->imageSize(imageSize)");
        }
        NeuralNet *net = new NeuralNet(cl, _numPlanes, _imageSize);
        delete this;
        return net;
    } else {
        NeuralNet *net = new NeuralNet(cl);
        delete this;
        return net;
    }
}

