


// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "../../EasyCL/util/StatefulTimer.h"

#include "../layer/LayerMaker.h"
#include "SoftMaxLayerPredict.h"


using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

SoftMaxLayerPredict::SoftMaxLayerPredict(Layer *previousLayer, SoftMaxMaker *maker, int batch) :
    LossLayer(previousLayer, maker),
        perPlane(maker->_perPlane),
        imageSize(previousLayer->getOutputSize()),
        numPlanes(previousLayer->getOutputPlanes()),
        imageSizeSquared(previousLayer->getOutputSize() * previousLayer->getOutputSize()),
        output(0),
        gradInput(0),
        allocatedSize(0),
        batchSize(batch),
        cl(maker->cl)
         {


}
VIRTUAL SoftMaxLayerPredict::~SoftMaxLayerPredict() {
    if(gradInput != 0) {
        delete[] gradInput;
    }
    if(output != 0) {
        delete[] output;
    }
}
VIRTUAL std::string SoftMaxLayerPredict::getClassName() const {
    return "SoftMaxLayerPredict";
}
VIRTUAL float *SoftMaxLayerPredict::getOutput() {
    return output;
}
VIRTUAL float *SoftMaxLayerPredict::getGradInput() {
    return gradInput;
}
VIRTUAL void SoftMaxLayerPredict::setBatchSize(int batchSize) {
    this->batchSize = batchSize;
    if(batchSize <= this->allocatedSize) {
        return;
    }
    if(output != 0) {
        delete[] output;
    }
    if(gradInput != 0) {
        delete[] gradInput;
    }
    output = new float[ getOutputNumElements() ];
    gradInput = new float[ previousLayer-> getOutputNumElements() ];
    allocatedSize = batchSize;
}
VIRTUAL int SoftMaxLayerPredict::getBatchSize() {
    return this->batchSize;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayerPredict::calcLossFromLabels(int const *labels) {
//    cout << "SoftMaxLayerPredict::calcloss" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcLossfromlabels");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int label = labels[n * numPlanes + plane];
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                loss += - log(output[ imageOffset + label ]);
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            loss += - log(output[imageOffset + label]);
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayerPredict calcLossfromlabels");
    return loss;
}
// need to calculate multinomial logistic /cross-entropy loss
VIRTUAL float SoftMaxLayerPredict::calcLoss(float const *expectedValues) {
    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcLoss");
    float loss = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    if(expectedValues[ imageOffset + i ] != 0) {
                        float thisloss = - expectedValues[ imageOffset + i ] * log(output[ imageOffset + i ]);
                        loss += thisloss;
                    }
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                float thisloss = - expectedValues[imageOffset + plane] * log(output[imageOffset + plane]);
                loss += thisloss;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayerPredict calcLoss");
    return loss;
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayerPredict::calcGradInputFromLabels(int const *labels) {
//    cout << "SoftMaxLayerPredict::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcGradInputfromlabels");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                int label = labels[n * numPlanes + plane];
                for(int i = 0; i < imageSizeSquared; i++) {
                    gradInput[imageOffset + i] = output[imageOffset + i];
                }
                gradInput[imageOffset + label] -= 1;
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            for(int plane = 0; plane < numPlanes; plane++) {
                gradInput[imageOffset + plane] = output[imageOffset + plane];
            }
            if(label >= numPlanes) {
                throw runtime_error("Label " + toString(label) + " exceeds number of softmax planes " + toString(numPlanes) );
            } else if(label < 0) {
                throw runtime_error("Label " + toString(label) + " negative");
            }
            gradInput[imageOffset + label] -= 1;
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayerPredict calcGradInputfromlabels");
}
// calculate partial deriv loss wrt our inputs, in other words, product of
// (multinomial cross-entropy) loss derivative wrt our output, and
// derivative of softmax wrt our inputs
VIRTUAL void SoftMaxLayerPredict::calcGradInput(float const *expectedValues) {
//    cout << "SoftMaxLayerPredict::calcerrors" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcGradInput");
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                for(int i = 0; i < imageSizeSquared; i++) {
                    int resultIndex = imageOffset + i;
                    gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            for(int plane = 0; plane < numPlanes; plane++) {
                int resultIndex = imageOffset + plane;
                gradInput[resultIndex] = output[resultIndex] - expectedValues[resultIndex];
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayerPredict calcGradInput");
}
VIRTUAL int SoftMaxLayerPredict::getNumLabelsPerExample() {
    if(perPlane) {
        return numPlanes;
    } else {
        return imageSizeSquared;
    }
}
VIRTUAL int SoftMaxLayerPredict::getPersistSize(int version) const {
    return 0;
}
VIRTUAL int SoftMaxLayerPredict::calcNumRightFromLabels(int const*labels) {
    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcNumRight");
//    float *input = previousLayer->getOutput(); // just retrieve as host-side array for now
    int numRight = 0;
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                int label = labels[n * numPlanes + plane];
                float thisMax = output[imageOffset + 0];
                int iMax = 0;
                for(int i = 1; i < imageSizeSquared; i++) {
                    if(output[imageOffset + i] > thisMax) {
                        thisMax = output[imageOffset + i];
                        iMax = i;
                    }
                }
                if(label == iMax) {
//                    cout << "n " << n << " plane " << plane << " label " << label << endl;
                    numRight++;
                }
            }
        }
    } else {
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            int label = labels[n];
            float thisMax = output[imageOffset + 0];
            int iMax = 0;
            for(int i = 1; i < numPlanes; i++) {
                if(output[imageOffset + i] > thisMax) {
                    thisMax = output[imageOffset + i];
                    iMax = i;
                }
            }
            if(label == iMax) {
                numRight++;
            }
        }
    }

    StatefulTimer::timeCheck("start SoftMaxLayerPredict calcNumRight");
    return numRight;
}
// for forward, we just need to apply the softmax activation. "just" :-P
VIRTUAL void SoftMaxLayerPredict::forward() {
	LOGI("SoftMaxLayerPredict::forward()");
//    cout << "SoftMaxLayerPredict::forward" << endl;
    StatefulTimer::timeCheck("start SoftMaxLayerPredict forward");

    cl->finish();
    previousLayer->getOutputWrapper()->copyToHost();//olivier don t forget
    cl->finish();
    float *input = previousLayer->getOutput(); // just retrieve as host-side array for now
    if(perPlane) {
        for(int n = 0; n < batchSize; n++) {
            for(int plane = 0; plane < numPlanes; plane++) {
                int imageOffset = (n * numPlanes + plane) * imageSizeSquared;
                float maxValue = input[imageOffset + 0];
                for(int i = 1; i < imageSizeSquared; i++) {
                    maxValue = std::max(maxValue, input[imageOffset + i]);
                }
                float denominator = 0;
                for(int i = 0; i < imageSizeSquared; i++) {
                    denominator += exp(input[imageOffset + i] - maxValue);
                }
                for(int i = 0; i < imageSizeSquared; i++) {
                    output[imageOffset + i] = exp(input[imageOffset + i] - maxValue) / denominator;
                }
            }
        }
    } else {
    	LOGI("ICIIIII");
        // force imagesize of 1 for now
        if(imageSize != 1) {
            throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
        }
        for(int n = 0; n < batchSize; n++) {
            int imageOffset = n * numPlanes * imageSizeSquared;
            // first get the max
            float maxValue = input[imageOffset + 0]; // since we assume imagesize 1, this is correct
            for(int plane = 1; plane < numPlanes; plane++) {
                maxValue = std::max(maxValue, input[imageOffset + plane]);
            }
            // calculate sum, under this max
            float denominator = 0;
            for(int plane = 0; plane < numPlanes; plane++) {
                denominator += exp(input[imageOffset + plane] - maxValue);
            }
            // now calc the softmaxes:
            for(int plane = 0; plane < numPlanes; plane++) {
                output[imageOffset + plane] = exp(input[imageOffset + plane] - maxValue) / denominator;
            }
        }
    }
    StatefulTimer::timeCheck("end SoftMaxLayerPredict forward");
    LOGI("SoftMaxLayerPredict::end forward()");
}
VIRTUAL void SoftMaxLayerPredict::getLabels(int *labels) { // need to allocate labels array first, and have called 'forward' first
    if(perPlane) {
        throw std::runtime_error("getLabels doesnt work with 'perPlane' option currently, though it wouldnt be hard to add, so ask if you need");
    }
    if(imageSize != 1) {
        throw std::runtime_error("perColumn only supported for imagesize 1 for now.  Sit tight :-)  (But please raise an issue to highlight your need)");
    }
    for(int n = 0; n < batchSize; n++) {
        float *outputStack = output + n * numPlanes;
        float highestProb = outputStack[0];
        int bestPlane = 0;
        for(int plane = 1; plane < numPlanes; plane++) {
            if(outputStack[plane] > highestProb) {
                bestPlane = plane;
                highestProb = outputStack[plane];
            }
        }
        labels[n] = bestPlane;

        if (n==0){
        	LOGI("prob %f",highestProb);
        }
    }
}
// this seems to be handled by calcGradInput? So, just to a nop?
// (cos this layer kind of combines loss layer and a 'normal' propagation layer)
// certainly, we dont have any weights to update, and we already handled error
// propagation in 'calcGradInput' method above
//VIRTUAL void SoftMaxLayerPredict::backward(float learningRate) {
//    cout << "SoftMaxLayerPredict::backproperrors" << endl;
    // nop, do nothing :-)
//}
VIRTUAL std::string SoftMaxLayerPredict::asString() const {
    return "SoftMaxLayerPredict{ perPlane=" + toString(perPlane) + " numPlanes=" + toString(numPlanes)
        + " imageSize=" + toString(imageSize) + " }";
}

CLWrapper * SoftMaxLayerPredict::getLossWrapper(){
	return 0;
}

CLWrapper * SoftMaxLayerPredict::getNbRightWrapper(){
	return 0;
}


