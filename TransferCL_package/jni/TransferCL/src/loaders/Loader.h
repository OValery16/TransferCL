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

#define LIBJPEG_FOUND

#define VIRTUAL virtual
#define STATIC static

class Loader {
    public:
    VIRTUAL std::string getType() = 0;
    VIRTUAL void load(unsigned char *data, int *labels, int startRecord, int numRecords) = 0;
    VIRTUAL int getImageCubeSize() = 0;
    VIRTUAL int getN() = 0;
    VIRTUAL int getPlanes() = 0;
    VIRTUAL int getImageSize() = 0;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    // [[[end]]]
};

