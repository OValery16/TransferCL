// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <stdexcept>
#include <string>
#include <iostream>
#include <algorithm>

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

// passed to trainers, for each training batch
class TransferCL_EXPORT TrainingContext {
public:
    const int epoch; // zero-based
    const int batch;
    // could add in other things here, like batch number etc...

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.add()
    // ]]]
    // generated, using cog:
    TrainingContext(int epoch, int batch);
    int getEpoch();
    int getBatch();

    // [[[end]]]
};

