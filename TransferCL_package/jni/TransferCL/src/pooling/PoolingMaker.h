// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#pragma once

#include "../dependencies.h"
#include "../layer/LayerMaker.h"
#include "../TransferCLDllExport.h"

/// \brief Use to create a Max-Pooling layer
///
/// Stride is fixed to equal the pooling size, so these are 
/// non-overlapping pools
PUBLICAPI
class TransferCL_EXPORT PoolingMaker : public LayerMaker2 {
public:
//    Layer *previousLayer;
    int _poolingSize;
    bool _padZeros;
    PUBLICAPI PoolingMaker() :
        _poolingSize(2),
        _padZeros(false) {
    }
    PUBLICAPI PoolingMaker *poolingSize(int _poolingSize) {

        this->_poolingSize = _poolingSize;
        return this;
    }
    PoolingMaker *padZeros() {
        this->_padZeros = true;
        return this;
    }
    PUBLICAPI static PoolingMaker *instance() {
        return new PoolingMaker();
    }
    virtual PoolingMaker *clone() const {
        return new PoolingMaker(*this);
    }
    virtual Layer *createLayer(Layer *previousLayer);
};


