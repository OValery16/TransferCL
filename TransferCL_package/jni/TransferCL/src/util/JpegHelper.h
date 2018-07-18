// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <string>

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

class TransferCL_EXPORT JpegHelper {

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    STATIC void write(std::string filename, int planes, int width, int height, unsigned char *values);
    STATIC void read(std::string filename, int planes, int width, int height, unsigned char *values);

    // [[[end]]]
};

