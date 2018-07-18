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
// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>

#include "Loader.h"

#define VIRTUAL virtual
#define STATIC static

class ManifestLoaderv1 : public Loader {
    private:
    bool includeLabels;
    std::string imagesFilepath;
    int N;
    int planes;
    int size;

    std::string *files;
    int *labels;

    // [[[cog
    // import cog_addheaders
    // cog_addheaders.addv2()
    // ]]]
    // generated, using cog:

    public:
    ~ManifestLoaderv1();
    STATIC bool isFormatFor(std::string imagesFilepath);
    ManifestLoaderv1(std::string imagesFilepath);
    ManifestLoaderv1(std::string imagesFilepath, bool includeLabels);
    VIRTUAL std::string getType();
    VIRTUAL int getImageCubeSize();
    VIRTUAL int getN();
    VIRTUAL int getPlanes();
    VIRTUAL int getImageSize();
    VIRTUAL void load(unsigned char *data, int *labels, int startRecord, int numRecords);

    private:
    void init(std::string imagesFilepath, bool includeLabels);
    int readIntValue(std::vector< std::string > splitLine, std::string key);

    // [[[end]]]
};

