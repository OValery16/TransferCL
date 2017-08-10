// Copyright Hugh Perkins 2015 hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <string>

#include "../DeepCL/src/net/NeuralNet.h"
#include "../DeepCL/src/layer/LayerMakers.h"
#include "../DeepCL/src/util/stringhelper.h"
#include "NetdefToCompile.h"
#include "../DeepCL/src/activate/ActivationFunction.h"
#include "../DeepCL/src/weights/WeightsInitializer.h"

using namespace std;



// string is structured like:
// prefix-nn*(inner)-postfix
// or:
// prefix-nn*inner-postfix
static std::string expandMultipliers(std::string netdef) {



    int starPos = netdef.find("*");
    if(starPos != (int)string::npos) {



        int prefixEnd = netdef.rfind("-", starPos);
        string prefix = "";
        string nnString = "";
        if(prefixEnd == (int)string::npos) {
            prefixEnd = -1;
            nnString = netdef.substr(0, starPos);
        } else {
            prefixEnd--;
            prefix = netdef.substr(0, prefixEnd + 1);
            cout << "prefix: [" << prefix << "]" << endl;
            nnString = netdef.substr(prefixEnd + 2, starPos - prefixEnd - 2);
        }
        cout << "nnString: [" << nnString << "]" << endl;
        int repeatNum = atoi(nnString);
        cout << "repeatNum " << repeatNum << endl;
        string remainderString = netdef.substr(starPos + 1);
        cout << "remainderString [" << remainderString << "]" << endl;
        string inner = "";
        string postfix = "";
        if(remainderString.substr(0, 1) == "(") {
            // need to find other ')', assume not nested for now...
            int rhBracket = remainderString.find(")");
            if(rhBracket == (int)string::npos) {

                throw runtime_error("matching bracket not found in " + remainderString);
//                return false;
            }
            inner = remainderString.substr(1, rhBracket - 1);
            cout << "inner [" << inner << "]" << endl;
            string newRemainder = remainderString.substr(rhBracket + 1);
            cout << "newRemainder [" << newRemainder << "]" << endl;
            if(newRemainder != "") {
                if(newRemainder[0] != '-') {
                    throw runtime_error("expect '-' after ')' in " + remainderString);
    //                return false;
                }
                postfix = newRemainder.substr(1);
                cout << "postfix [" << postfix << "]" << endl;
            }
        } else {
            int innerEnd = remainderString.find("-");
            if(innerEnd == (int)string::npos) {
                innerEnd = remainderString.length();
            } else {
//                innerEnd;
                postfix = remainderString.substr(innerEnd + 1);
                cout << "postfix [" << postfix << "]" << endl;
            }
            inner = remainderString.substr(0, innerEnd);
            cout << "inner [" << inner << "]" << endl;
//            if(remainderString.find("-") != string::npos) {
//                sectionEndPos = remainderString.find("-");
//            }
        }
//        return "";
        // if remainderString starts with (, then repeat up to next)
        // otherwise, repeat up to next -
//        int sectionEndPos = remainderString.length();
//        remainderString = 
        string newString = prefix;
        for(int i = 0; i < repeatNum; i++) {
            if(newString != "") {
                newString += "-";
            }
            newString += expandMultipliers(inner);
        }
        if(postfix != "") {
            newString += "-" + expandMultipliers(postfix);
        }
        cout << "multiplied string: " << newString << endl;
        return newString;
    } else {
        return netdef;
    }    
}
bool NetdefToCompile::createNetForCompile(int batchsize, NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer) {

	//netdef=expandMultipliers(netdef);
	LOGI("netdef %s",netdef.c_str());
	vector<string> splitConfigNetDef = split(netdef, "-");
	for(int i = 0; i < (int)splitConfigNetDef.size(); i++) {
	   LOGI("splitNetDef");
	   string layerDef = splitConfigNetDef[i];
	   if(layerDef.find("c") != string::npos){
		  // string layerDef2 ="linear";
		   string layerDef2="";
		   string layerDef3="";
		   if ((int)splitConfigNetDef.size()-i-1 !=0)
			   layerDef2 = splitConfigNetDef[i+1];
		   else
			   layerDef2="linear";

		   if ((int)splitConfigNetDef.size()-i-2 !=0)
			   layerDef3 = splitConfigNetDef[i+2];
		   compileConv(batchsize,net, layerDef,layerDef2,layerDef3, weightsInitializer);
	   }
	   else if(layerDef.find("mp") != string::npos)
		   compileMaxPooling(batchsize,net, layerDef, weightsInitializer);
	   else if(layerDef.find("drop") != string::npos)
		   compileDropout(batchsize,net, layerDef, weightsInitializer);
	   else if((layerDef.find("relu") != string::npos)||(layerDef.find("elu") != string::npos)||(layerDef.find("tanh") != string::npos)||(layerDef.find("sigmoid") != string::npos)||(layerDef.find("linear") != string::npos))
		   compileActivation(batchsize,net, layerDef, weightsInitializer);
	   else if(layerDef.find("rt") != string::npos)
		   compileRandomTranslation(batchsize,net, layerDef, weightsInitializer);
	   else if(layerDef.find("n") != string::npos){
		   string layerDef2 ="linear";
//		   if ((i+1)<(int)splitConfigNetDef.size())
//			   layerDef2 = splitConfigNetDef[i+1];
		   compileFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer);
	   }
	}


    net->addLayer(SoftMaxMaker::instance());
    return true;
}

void NetdefToCompile::compileConv(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,std::string pooling_layerDef, WeightsInitializer *weightsInitializer) {

		bool useMaxPooling=false;
		int poolingSize =0;
		if (pooling_layerDef.length()!=0){
			poolingSize=atoi(pooling_layerDef.substr(pooling_layerDef.find("mp")+2));
			useMaxPooling=true;
		}

	    int positionS=layerDef.find("s");
		int positionC=layerDef.find("c");
		int positionZ=layerDef.find("z");


		int stride = atoi(layerDef.substr(0,positionS));
        int numFilters = atoi(layerDef.substr(positionS+1,positionC));

        int filterSize =1;
        int padZeros = 0;
        if (positionZ!=std::string::npos){
        	filterSize = atoi(layerDef.substr(positionC+1,positionZ-1));
        	padZeros = 1;
        }else{
            filterSize = atoi(layerDef.substr(positionC+1));
            padZeros = 0;
        }
        int skip = 0;

        net->addLayer(ConvolutionalMaker::instance()->stride(stride)->numFilters(numFilters)->filterSize(filterSize)->padZeros(padZeros)->biased()->weightsInitializer(weightsInitializer)->batchSize(batchsize)->activationLayer(activation_layerDef)->useMaxPooling(useMaxPooling)->maxPool_spatialExtent(poolingSize)->maxPool_strides(poolingSize) );

}

void NetdefToCompile::compileMaxPooling(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {

	int poolingSize = atoi(layerDef.substr(layerDef.find("mp")+2));
	LOGI("Pooling size %d",poolingSize);
	net->addLayer(PoolingMaker::instance()->poolingSize(poolingSize));
}

void NetdefToCompile::compileDropout(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {
	net->addLayer(DropoutMaker::instance()->dropRatio(0.5f));
}

void NetdefToCompile::compileActivation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {

	if(layerDef.find("relu") != string::npos) {
	    	LOGI("relu");
	        net->addLayer(ActivationMaker::instance()->relu());
	} else if(layerDef.find("elu") != string::npos) {
	    	LOGI("elu");
	        net->addLayer(ActivationMaker::instance()->elu());
	} else if(layerDef.find("tanh") != string::npos) {
	    	LOGI("tanh");
	        net->addLayer(ActivationMaker::instance()->tanh());
	} else if(layerDef.find("sigmoid") != string::npos) {
	    	LOGI("sigmoid");
	        net->addLayer(ActivationMaker::instance()->sigmoid());
	} else if(layerDef.find("linear") != string::npos) {
	    	LOGI("linear");
	        net->addLayer(ActivationMaker::instance()->linear()); // kind of pointless nop, but useful for testing
	}
}

void NetdefToCompile::compileRandomTranslation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {
	LOGI("RandomTranslationsMaker");
	int translateSize = atoi(layerDef.substr(layerDef.find("rt")+2));
	net->addLayer(RandomTranslationsMaker::instance()->translateSize(translateSize) );
}

void NetdefToCompile::compileFullyConnectedLayer(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,  WeightsInitializer *weightsInitializer) {

	//vector<string> fullDef = split(baseLayerDef, "n");
	int positionN=layerDef.find("n");
	int numPlanes = atoi(layerDef.substr(0,positionN));
	int biased=1;
	LOGI("FullyConnectedMaker");
	net->addLayer(FullyConnectedMaker::instance()->numPlanes(numPlanes)->imageSize(1)->biased(biased)->weightsInitializer(weightsInitializer)->batchSize(batchsize)->activationLayer(activation_layerDef) );

}


