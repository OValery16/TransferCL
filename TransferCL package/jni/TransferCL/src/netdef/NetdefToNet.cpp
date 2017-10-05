// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <iostream>
#include <string>

#include "../net/NeuralNet.h"
#include "../layer/LayerMakers.h"
#include "../util/stringhelper.h"
#include "NetdefToNet.h"
#include "../activate/ActivationFunction.h"
#include "../weights/WeightsInitializer.h"

using namespace std;

#undef STATIC
#undef VIRTUAL
#define STATIC
#define VIRTUAL

// string is structured like:
// prefix-nn*(inner)-postfix
// or:
// prefix-nn*inner-postfix
STATIC std::string expandMultipliers(std::string netdef) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/netdef/NetdefToNet.cpp: string expandMultipliers");
#endif


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

            }
            inner = remainderString.substr(1, rhBracket - 1);
            cout << "inner [" << inner << "]" << endl;
            string newRemainder = remainderString.substr(rhBracket + 1);
            cout << "newRemainder [" << newRemainder << "]" << endl;
            if(newRemainder != "") {
                if(newRemainder[0] != '-') {
                    throw runtime_error("expect '-' after ')' in " + remainderString);
                }
                postfix = newRemainder.substr(1);
                cout << "postfix [" << postfix << "]" << endl;
            }
        } else {
            int innerEnd = remainderString.find("-");
            if(innerEnd == (int)string::npos) {

                innerEnd = remainderString.length();
            } else {
                postfix = remainderString.substr(innerEnd + 1);
                cout << "postfix [" << postfix << "]" << endl;
            }
            inner = remainderString.substr(0, innerEnd);
            cout << "inner [" << inner << "]" << endl;

        }

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

PUBLICAPI STATIC bool NetdefToNet::createNetFromNetdef(int batchsize,NeuralNet *net, std::string netdef) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/netdef/NetdefToNet.cpp: createNetFromNetdef");
#endif


    OriginalInitializer originalInitializer;
    return createNetFromNetdef(batchsize,net, netdef, &originalInitializer);
}
PUBLICAPI STATIC bool NetdefToNet::createNetFromNetdefCharStar(int batchsize,NeuralNet *net, const char *netdef) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/netdef/NetdefToNet.cpp: createNetFromNetdefCharStar");
#endif


    OriginalInitializer originalInitializer;
    return createNetFromNetdef(batchsize,net, netdef, &originalInitializer);
}

STATIC bool NetdefToNet::createNetFromNetdef(int batchsize, NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer) {

	//netdef=expandMultipliers(netdef);
	//LOGI("------------------create network %s",netdef.c_str());
	vector<string> splitConfigNetDef = split(netdef, "-");
	for(int i = 0; i < (int)splitConfigNetDef.size(); i++) {
	   //LOGI("-------------------- splitNetDef layer %d ",i);
	   string layerDef = splitConfigNetDef[i];

	   if(layerDef.find("c") != string::npos){
		   string layerDef2="";
		   string layerDef3="";

		   layerDef2="linear";
		   layerDef3="no maxpool";
		   if ((int)splitConfigNetDef.size()-i-1 !=0){
			   if((splitConfigNetDef[i+1].find("relu") != string::npos)||(splitConfigNetDef[i+1].find("elu") != string::npos)||(splitConfigNetDef[i+1].find("tanh") != string::npos)||(splitConfigNetDef[i+1].find("sigmoid") != string::npos)||(splitConfigNetDef[i+1].find("linear") != string::npos)){
				   layerDef2 = splitConfigNetDef[i+1];
			   }else{
				   if(splitConfigNetDef[i+1].find("mp") != string::npos){
					   layerDef3 = splitConfigNetDef[i+1];
				   }
			   }
		   }
		   if ((int)splitConfigNetDef.size()-i-2 !=0){
			   if(splitConfigNetDef[i+2].find("mp") != string::npos){
			   		layerDef3 = splitConfigNetDef[i+2];
			   }
		   }

		   ////////////
		   createConv(batchsize,net, layerDef,layerDef2,layerDef3, weightsInitializer);
	   }
	   else if(layerDef.find("mp") != string::npos)
		   createMaxPooling(batchsize,net, layerDef, weightsInitializer);
	   else if((layerDef.find("relu") != string::npos)||(layerDef.find("elu") != string::npos)||(layerDef.find("tanh") != string::npos)||(layerDef.find("sigmoid") != string::npos)||(layerDef.find("linear") != string::npos))
		   createActivation(batchsize,net, layerDef, weightsInitializer);
	   else if(layerDef.find("n") != string::npos){
		   string layerDef2 ="linear";
		   if ((i+1)<(int)splitConfigNetDef.size())
			   layerDef2 = splitConfigNetDef[i+1];
		   if(i==(splitConfigNetDef.size()-1))
			   createFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer,true);
		   else
			   createFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer,false);
	   }
	}


    net->addLayer(SoftMaxMaker::instance(false,batchsize));
    return true;
}

STATIC bool NetdefToNet::createNetFromNetdefPrediction(int batchsize, NeuralNet *net, std::string netdef, WeightsInitializer *weightsInitializer) {

	//netdef=expandMultipliers(netdef);
	//LOGI("netdef %s",netdef.c_str());
	vector<string> splitConfigNetDef = split(netdef, "-");
	for(int i = 0; i < (int)splitConfigNetDef.size(); i++) {
		//LOGI("----- LAYER %s",splitConfigNetDef[i].c_str());
	   string layerDef = splitConfigNetDef[i];
	   if(layerDef.find("c") != string::npos){
		  // string layerDef2 ="linear";
		   string layerDef2="";
		   string layerDef3="";
		   if ((int)splitConfigNetDef.size()-i-1 !=0){
			   layerDef2 = splitConfigNetDef[i+1];
		   	   if ((int)splitConfigNetDef.size()-i-2 !=0)
		   		  layerDef3 = splitConfigNetDef[i+2];
		   	   else
		   		  layerDef3="no maxpool";
		   }else{
			   layerDef2="linear";
		   	   layerDef3="no maxpool";
		   }
		   createConv(batchsize,net, layerDef,layerDef2,layerDef3, weightsInitializer);
	   }
	   else if(layerDef.find("mp") != string::npos)
		   createMaxPooling(batchsize,net, layerDef, weightsInitializer);
	   else if((layerDef.find("relu") != string::npos)||(layerDef.find("elu") != string::npos)||(layerDef.find("tanh") != string::npos)||(layerDef.find("sigmoid") != string::npos)||(layerDef.find("linear") != string::npos))
		   createActivation(batchsize,net, layerDef, weightsInitializer);
	   else if(layerDef.find("n") != string::npos){
		   string layerDef2 ="linear";
		   if ((i+1)<(int)splitConfigNetDef.size())
			   layerDef2 = splitConfigNetDef[i+1];

		   if(i==(splitConfigNetDef.size()-1))
		   			   createFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer,true);
		   		   else
		   			   createFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer,false);
		   //createFullyConnectedLayer(batchsize,net, layerDef,layerDef2, weightsInitializer);
	   }
	}


    net->addLayer(SoftMaxMaker::instance(false,batchsize));
    return true;
}
void NetdefToNet::createConv(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,std::string pooling_layerDef, WeightsInitializer *weightsInitializer) {


		bool useMaxPoolingTemp=false;
		int poolingSize =0;
		if (pooling_layerDef.length()!=0){

			std::stringstream stream;
				stream <<pooling_layerDef.substr(pooling_layerDef.find("mp")+2);
				if (!(stream >>poolingSize)){
					useMaxPoolingTemp=false;
				}else
					useMaxPoolingTemp=true;
		}
//		LOGI("poolingSize %d useMaxPoolingTemp %d",poolingSize, useMaxPoolingTemp);
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
        //useMaxPoolingTemp=false;
        net->addLayer(ConvolutionalMaker::instance()->stride(stride)->numFilters(numFilters)->filterSize(filterSize)->padZeros(padZeros)->biased()->weightsInitializer(weightsInitializer)->batchSize(batchsize)->activationLayer(activation_layerDef)->useMaxPooling(useMaxPoolingTemp)->maxPool_spatialExtent(poolingSize)->maxPool_strides(poolingSize) );

}

void NetdefToNet::createMaxPooling(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {

	int poolingSize = atoi(layerDef.substr(layerDef.find("mp")+2));
	//LOGI("-----------Pooling size %d",poolingSize);
	net->addLayer(PoolingMaker::instance()->poolingSize(poolingSize));
}

void NetdefToNet::createActivation(int batchsize,NeuralNet *net, std::string layerDef, WeightsInitializer *weightsInitializer) {

	if(layerDef.find("relu") != string::npos) {
	    	//LOGI("-----------relu");
	        net->addLayer(ActivationMaker::instance()->relu());
	} else if(layerDef.find("elu") != string::npos) {
	    	//LOGI("-----------elu");
	        net->addLayer(ActivationMaker::instance()->elu());
	} else if(layerDef.find("tanh") != string::npos) {
	    	//LOGI("-----------tanh");
	        net->addLayer(ActivationMaker::instance()->tanh());
	} else if(layerDef.find("sigmoid") != string::npos) {
	    	//LOGI("-----------sigmoid");
	        net->addLayer(ActivationMaker::instance()->sigmoid());
	} else if(layerDef.find("linear") != string::npos) {
	    	//LOGI("-----------linear");
	        net->addLayer(ActivationMaker::instance()->linear());
	}
}


void NetdefToNet::createFullyConnectedLayer(int batchsize,NeuralNet *net, std::string layerDef, std::string activation_layerDef,  WeightsInitializer *weightsInitializer,bool isLastB) {

	//vector<string> fullDef = split(baseLayerDef, "n");
	int positionN=layerDef.find("n");
	int numPlanes = atoi(layerDef.substr(0,positionN));
	int biased=1;
	//LOGI("FullyConnectedMaker");
	net->addLayer(FullyConnectedMaker::instance()->numPlanes(numPlanes)->imageSize(1)->biased(biased)->weightsInitializer(weightsInitializer)->batchSize(batchsize)->activationLayer(activation_layerDef)->isLast(isLastB));

}



