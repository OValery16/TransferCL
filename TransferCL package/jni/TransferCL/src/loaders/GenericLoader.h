// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include <iostream>
#include <string>
#include <algorithm>
#include <stdexcept>

#include "../TransferCLDllExport.h"

#define VIRTUAL virtual
#define STATIC static

/// \brief Use to load data from file, given the path to the images file
///
/// Can handle mnist, norb and kgsgov2 formats for now
/// Can be extended to other formats, as long as there is some
/// reasonably quick way to determine the format correctly
/// eg, a header, or based on the file extension
PUBLICAPI
class TransferCL_EXPORT GenericLoader {

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    PUBLICAPI STATIC void getDimensions(const char * trainFilepath, int *p_numExamples, int *p_numPlanes, int *p_imageSize);
    PUBLICAPI STATIC void load(const char * imagesFilePath, float *images, int *labels, int startN, int numExamples);
    STATIC void load(const char * trainFilepath, unsigned char *images, int *labels);
    STATIC void load(const char * trainFilepath, unsigned char *images, int *labels, int startN, int numExamples);

    // [[[end]]]
};

