// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <algorithm>
#include <iostream>
#include <stdexcept>

class NeuralNet;
class Trainable;

#define VIRTUAL virtual
#define STATIC static

#include "../TransferCLDllExport.h"

// this handles learning one single epoch, breaking up the incoming training or testing
// data into batches, which are then sent to the NeuralNet for forward and backward
// propagation.
//class TransferCL_EXPORT BatchLearnerOnDemand {
//public:
//    Trainable *net; // NOT owned by us, dont delete

//    // [[[cog
//    // import cog_addheaders
//    // cog_addheaders.add()
//    // ]]]
// generated, using cog:

//    // [[[end]]]
//};

