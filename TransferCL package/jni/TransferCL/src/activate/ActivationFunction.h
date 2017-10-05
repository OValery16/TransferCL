// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "../dependencies.h"
#include <string>
#include <stdexcept>
#include <cmath>

#include "../TransferCLDllExport.h"

class TransferCL_EXPORT ActivationFunction {
public:
    virtual ~ActivationFunction() {}
    virtual float calc(float value) const { throw std::runtime_error("calc not implemented"); };
    virtual float calcDerivative(float output) const { throw std::runtime_error("calcDerivative not implemented"); };
    virtual float getFalse() const {  throw std::runtime_error("getFalse not implemented"); } 
    virtual float getTrue() const {  throw std::runtime_error("getTrue not implemented"); } 
    virtual const char * getDefineName() const { throw std::runtime_error("getDefineName not implemented"); } 
    static ActivationFunction *fromName(std::string name);
};

class TanhActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return tanh(value);
    }
    virtual float calcDerivative(float output) const {
        return 1 - output * output;
    }
    virtual float getTrue() const {
        return 0.5f;
    }
    virtual float getFalse() const {
        return -0.5f;
    }
    virtual const char * getDefineName() const {
        return "TANH";
    } 
};

class ScaledTanhActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return 1.7159f * tanh(value * 0.66667f);
    }
    virtual float calcDerivative(float output) const {
        return 0.66667f * (1.7159f - 1 / 1.7159f * output * output);
    }
    virtual float getTrue() const {
        return 1.0f;
    }
    virtual float getFalse() const {
        return -1.0f;
    }
    virtual const char * getDefineName() const {
        return "SCALEDTANH";
    } 
};

class SigmoidActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return 1.0f / (1.0f + exp(- value) );
    }
    virtual float calcDerivative(float output) const {
        return output * (1 - output);
    }
    virtual float getTrue() const {
        return 0.8f;
    }
    virtual float getFalse() const {
        return 0.2f;
    }
    virtual const char * getDefineName() const {
        return "SIGMOID";
    } 
};

class LinearActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return value;
    }
    virtual float calcDerivative(float output) const {
        return 1;
    }
    virtual float getTrue() const {
        return 0.5f;
    }
    virtual float getFalse() const {
        return -0.5f;
    }
    virtual const char * getDefineName() const {
        return "LINEAR";
    } 
};

class ReluActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return value > 0 ? value : 0;
    }
    virtual float calcDerivative(float output) const {
        return output > 0 ? 1.0f : 0.0f;
    }
    virtual float getTrue() const {
        return 0.8f;
    }
    virtual float getFalse() const {
        return 0.2f;
    }
    virtual const char * getDefineName() const {
        return "RELU";
    } 
};


class EluActivation : public ActivationFunction {
public:
    virtual float calc(float value) const {
        return value > 0 ? value : exp(value) - 1;
    }
    virtual float calcDerivative(float output) const {
        return output > 0 ? 1.0f : output + 1;
    }
    virtual float getTrue() const {
        return 0.8f;
    }
    virtual float getFalse() const {
        return 0.2f;
    }
    virtual const char * getDefineName() const {
        return "ELU";
    } 
};

