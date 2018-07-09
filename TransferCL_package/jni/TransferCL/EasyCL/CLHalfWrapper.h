// Copyright Olivier Valery 2016
// adapted from Hugh Perkins' version in order to add half support
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.
#pragma once

#include <stdexcept>

#include "EasyCL.h"
#include "CLWrapper.h"

//olivier: no half in this framework
//#include <CLBlast/include/clblast_half.h>


class CLHalfWrapper : public CLWrapper {
protected:
    half *hostarray;  // NOT owned by this object, do NOT free :-)
public:
    CLHalfWrapper(int N, half *_hostarray, EasyCL *easycl) :
             CLWrapper(N, easycl),
             hostarray(_hostarray)
              {
    }
    CLHalfWrapper(const CLHalfWrapper &source) :
        CLWrapper(0, 0), hostarray(0) { // copy constructor
        throw std::runtime_error("can't assign these...");
    }
    CLHalfWrapper &operator=(const CLHalfWrapper &two) { // assignment operator
       if(this == &two) { // self-assignment
          return *this;
       }
       throw std::runtime_error("can't assign these...");
    }
    inline half get(int index) {
        return hostarray[index];
    }
    virtual ~CLHalfWrapper() {
    }
    virtual int getElementSize() {
    	LOGI("getElementSize %d",sizeof(hostarray[0]));
        return sizeof(hostarray[0]);
    }
    virtual void *getHostArray() {
        return hostarray;
    }
    virtual void const*getHostArrayConst() {
        return hostarray;
    }
};



