// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../../EasyCL/EasyCL.h"

class OutputData;

class IAcceptsLabels {
public:
    virtual float calcLossFromLabels(int const*labels) = 0;
    virtual void calcGradInputFromLabels(int const*labels) = 0;
    virtual int calcNumRightFromLabels(int const*labels) = 0;
    virtual int getNumLabelsPerExample() = 0;
    virtual CLWrapper * getLossWrapper() = 0;
    virtual CLWrapper * getNbRightWrapper() = 0;
};

#include "../dependencies.h"

