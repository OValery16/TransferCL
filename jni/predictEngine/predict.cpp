// Copyright Hugh Perkins (hughperkins at gmail), Josef Moudrik 2015
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.


//#include "../DeepCL/src/DeepCL.h"
//#include "../DeepCL/src/loss/SoftMaxLayer.h"
#ifdef _WIN32
#include <stdio.h>
#include <fcntl.h>
#include <io.h>
#endif // _WIN32
//#include "../DeepCL/src/clblas/ClBlasInstance.h"

#include "predict.h"

using namespace std;



ConfigPrediction::ConfigPrediction() {
//default parameters

   gpuIndex = -1;
   weightsFile = "";//"/data/data/com.sony.openclexample1/preloadingData/weightstTransferedTEST.dat";//"/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";//"/data/data/com.sony.openclexample1/app_execdir/weightstface1.dat";//weightstodelete.dat";///weightstface1.dat";//weightstodelete.dat";//weightsTest13july.dat";//"weights.dat";
   batchSize = 128;
   inputFile = "";
   outputFile = "";
   outputLayer = -1;
   writeLabels = 1;
   outputFormat = "text";

}

PredictionModel::PredictionModel(){

//    if(config.gpuIndex >= 0) {
//        cl = EasyCL::createForIndexedGpu(config.gpuIndex);
//    } else {
        cl = EasyCL::createForFirstGpuOtherwiseCpu();
//    }

}

PredictionModel::~PredictionModel(){
	LOGI( "easyCL oject destroyed");
	delete cl;
	//LOGI( "easyCL oject destroyed done");

}
void PredictionModel::go0(ConfigPrediction config) {

	//LOGE( "je suis la0.");
	int N = -1;
	int numPlanes;
	int imageSize;
	int imageSizeCheck;
	GenericLoaderv2* loader = NULL;

	loader = new GenericLoaderv2(config.inputFile);
	N = loader->getN();
	numPlanes = loader->getPlanes();
	imageSize = loader->getImageSize();
	//LOGI("N=%d planes %d size %d",N,numPlanes,imageSize);
	const long inputCubeSize = numPlanes * imageSize * imageSize ;

	//LOGE( "je suis la0.5");
	NeuralNet *net;
	net = new NeuralNet(cl);
	WeightsInitializer *weightsInitializer = new OriginalInitializer();

	string netDef("default netDef");
	WeightsPersister::loadConfigString(config.weightsFile, netDef);

	net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
	net->addLayer(NormalizationLayerMaker::instance()->translate(0.0f)->scale(1.0f) ); // This will be read from weights file

	NetdefToNet::createNetFromNetdef(config.batchSize,net, netDef, weightsInitializer);

	int ignI;
	float ignF;
	WeightsPersister::loadWeights(config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF,false);
	float *inputData = new float[ inputCubeSize * config.batchSize];

	int *labels = new int[config.batchSize];
	int n = 0;
	ostream *outFile = 0;
	if(config.outputFormat == "text") {
	    outFile = new ofstream(config.outputFile, ios::out);
	} else if(config.outputFormat == "binary") {
	    outFile = new ofstream(config.outputFile, ios::out | std::ios::binary);
	}
	//number of layers
	config.outputLayer = net->getNumLayers() - 1;
	// pass 0 for labels, and this will cause GenericLoader to simply not try to load any labels
	// now, after modifying GenericLoader to have this new behavior
	// GenericLoader::load(config.inputFile.c_str(), inputData, 0, n, config.batchSize);
	loader->load(inputData, 0, n, config.batchSize);

    makePredictions(n,net,inputData, config,outFile,labels,inputCubeSize, N, loader);

	delete outFile;
	delete[] inputData;
	delete[] labels;
	delete weightsInitializer;
	delete net;
	delete loader;

}

void PredictionModel::makePredictions(int n,NeuralNet *net,float *inputData,ConfigPrediction config,ostream *outFile,int *labels,const long inputCubeSize, int N, GenericLoaderv2* loader){

	bool more = true;
	while(more) {
	        // no point in forwarding through all, so forward through each, one by one
//	        if(config.outputLayer < 0 || config.outputLayer > net->getNumLayers()) {
//	        	//LOGE( "outputLayer should be the layer number of one of the layers in the network");
//	            throw runtime_error("outputLayer should be the layer number of one of the layers in the network");
//	        }
//	        //LOGE( "je suis la 1.35");
	        dynamic_cast<InputLayer *>(net->getLayer(0))->in(inputData);
	        //LOGE( "je suis la 1.36");
	        for(int layerId = 0; layerId <= config.outputLayer; layerId++) {
	            //olivier StatefulTimer::setPrefix("layer" + toString(layerId) + " ");
	            net->getLayer(layerId)->forward();
	            //olivier StatefulTimer::setPrefix("");

	        }
	        LOGE( "je suis la 1.4");
	        if(!config.writeLabels) {
	        	//LOGE( "je suis la 1.45");
	            if(config.outputFormat == "text") {
	            	//LOGE( "je suis la 1.46");
	                float const*output = net->getLayer(config.outputLayer)->getOutput();
	                const int numFields = net->getLayer(config.outputLayer)->getOutputCubeSize();
	                //LOGE( "je suis la 1.47");
	                for(int i = 0; i < config.batchSize; i++) {
	                    for(int f = 0; f < numFields; f++) {
	                        if(f > 0) {
	                            *outFile << " ";
	                        }
	                        *outFile << output[ i * numFields + f ];
	                    }
	                    *outFile << "\n";
	                }
	            } else {
	                outFile->write(reinterpret_cast<const char *>(net->getOutput()), net->getOutputNumElements() * 4 * config.batchSize);
	            }
	        } else {
	        	//LOGE( "je suis la 1.46");
	            SoftMaxLayer *softMaxLayer = dynamic_cast< SoftMaxLayer *>(net->getLayer(config.outputLayer) );
	            if(softMaxLayer == 0) {
	            	//LOGE( "must choose softmaxlayer, if want to output labels" );
	                //cout << "must choose softmaxlayer, if want to output labels" << endl;
	                return;
	            }
	        	//LOGE( "je suis la 1.47");
	            softMaxLayer->getLabels(labels);
	            if(config.outputFormat == "text") {
	                for(int i = 0; i < config.batchSize; i++) {
	                    *outFile << labels[i] << "\n";
	                }
	            } else {
	                outFile->write(reinterpret_cast< char * >(labels), config.batchSize * 4l);
	            }
	            outFile->flush();
	        }
	        //LOGE( "je suis la 1.48");
	        n += config.batchSize;
	        if(config.inputFile == "") {
	            cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
	            more = !cin.eof();
	        } else {
	            if(n + config.batchSize < N) {
	            	//LOGE( "je suis la 1.49");
	                // GenericLoader::load(config.inputFile.c_str(), inputData, 0, n, config.batchSize);
	                loader->load(inputData, 0, n, config.batchSize);
	            } else {
	            	//LOGE( "je suis la 1.5");
	                more = false;
	                if(n != N) {
	                    cout << "breaking prematurely, since file is not an exact multiple of batchsize, and we didnt handle this yet" << endl;
	                }
	            }
	        }
	    }
}


void PredictionModel::go(ConfigPrediction config) {

	LOGI("################################################");
	LOGI("###################Prediction###################");
	LOGI("################################################");


    int N = -1;
    int numPlanes;
    int imageSize;
    int imageSizeCheck;
    GenericLoaderv2* loader = NULL;
    if(config.inputFile == "") {
        int dims[3];
        cin.read(reinterpret_cast< char * >(dims), 3 * 4l);
        numPlanes = dims[0];
        imageSize = dims[1];
        imageSizeCheck = dims[2];
        if(imageSize != imageSizeCheck) {
        	LOGE("imageSize doesnt match imageSizeCheck, image not square");
            throw std::runtime_error("imageSize doesnt match imageSizeCheck, image not square");
        }
    } else {
        loader = new GenericLoaderv2(config.inputFile);
        N = loader->getN();
        numPlanes = loader->getPlanes();
        imageSize = loader->getImageSize();

    }
    config.batchSize=N;//correct this problem in future version
    const long inputCubeSize = numPlanes * imageSize * imageSize ;

    LOGI("------- Network Generation");
    NeuralNet *net;
    net = new NeuralNet(cl);

    WeightsInitializer *weightsInitializer = new OriginalInitializer();

    if(config.weightsFile == "") {
    	LOGE( "----------- FATAL ERROR: weightsFile not specified");
        return;
    }


    string netDef("default netDef");
    if (!WeightsPersister::loadConfigString(config.weightsFile, netDef) ){
    	LOGE( "----------- FATAL ERROR: Cannot load network definition from weightsFile.");
        return;
    }
    LOGI("-----------Network Layers Creation %s",netDef.c_str());

    net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));

    net->addLayer(NormalizationLayerMaker::instance()->translate(0.0)->scale(1.0) ); // This will be read from weights file

    if(!NetdefToNet::createNetFromNetdef(config.batchSize,net, netDef, weightsInitializer) ) {
        return;
    }

    int ignI;
    float ignF;

    LOGI("-----------Loading the weights");
    if(!WeightsPersister::loadWeights(config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF,false) ){
    	LOGE( "Cannot load network weights from weightsFile.");
        return;
    }

    net->setBatchSize(config.batchSize);

    float *inputData = new float[ inputCubeSize * config.batchSize];

    int *labels = new int[config.batchSize];
    int n = 0;
    bool more = true;
    ostream *outFile = 0;
    if(config.outputFile == "") {
        outFile = &cout;
    } else {
        if(config.outputFormat == "text") {
            outFile = new ofstream(config.outputFile, ios::out);
        } else if(config.outputFormat == "binary") {
            outFile = new ofstream(config.outputFile, ios::out | std::ios::binary);
        } else {
            throw runtime_error("outputFormat " + config.outputFormat + " not recognized");
        }
    }
    if(config.outputLayer == -1) {
        config.outputLayer = net->getNumLayers() - 1;
    }
    if(config.inputFile == "") {
        cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
        more = !cin.eof();
    } else {

    	loader->load(inputData, 0, n, config.batchSize);
    }


    LOGI("-----------Start prediction");
    while(more) {

        if(config.outputLayer < 0 || config.outputLayer > net->getNumLayers()) {
            throw runtime_error("outputLayer should be the layer number of one of the layers in the network");
        }
        dynamic_cast<InputLayer *>(net->getLayer(0))->in(inputData);

        for(int layerId = 0; layerId <= config.outputLayer; layerId++) {
            net->getLayer(layerId)->forward();
        }
        if(!config.writeLabels) {
            if(config.outputFormat == "text") {
                float const*output = net->getLayer(config.outputLayer)->getOutput();
                const int numFields = net->getLayer(config.outputLayer)->getOutputCubeSize();
                for(int i = 0; i < config.batchSize; i++) {
                    for(int f = 0; f < numFields; f++) {
                        if(f > 0) {
                            *outFile << " ";
                        }
                        *outFile << output[ i * numFields + f ];
                    }
                    *outFile << "\n";
                }
            } else {
                outFile->write(reinterpret_cast<const char *>(net->getOutput()), net->getOutputNumElements() * 4 * config.batchSize);
            }
        } else {
        	SoftMaxLayer *softMaxLayer = dynamic_cast< SoftMaxLayer *>(net->getLayer(config.outputLayer) );
            if(softMaxLayer == 0) {
            	LOGE( "----------- FATAL ERROR: cannot define the softmax layer");
                return;
            }
            softMaxLayer->getLabels(labels);
            if(config.outputFormat == "text") {
                for(int i = 0; i < config.batchSize; i++) {
                    *outFile << labels[i] << "\n";
                }
            } else {
                outFile->write(reinterpret_cast< char * >(labels), config.batchSize * 4l);
            }
            outFile->flush();
        }

        n += config.batchSize;
        if(config.inputFile == "") {
            cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
            more = !cin.eof();
        } else {
            if(n + config.batchSize <= N) {
                loader->load(inputData, 0, n, config.batchSize);
            } else {

                more = false;
                if(n != N) {
                    LOGI("breaking prematurely, since file is not an exact multiple of batchsize, and we didnt handle this yet");
                }
            }
        }
    }
    LOGI("--------- Prediction: done (prediction in %s)",config.outputFile.c_str());

    if(config.outputFile != "") {
        delete outFile;
    }
    LOGI("--------- End of ther prediction: Delete objects");
    if(loader != NULL) delete loader;
    delete[] inputData;
    delete[] labels;
    delete weightsInitializer;
    delete net;
}

void PredictionModel::printUsage(char *argv[], ConfigPrediction config) {
    cout << "Usage: " << argv[0] << " [key]=[value] [[key]=[value]] ..." << endl;
    cout << endl;
    cout << "Possible key=value pairs:" << endl;
    /* [[[cog
        cog.outl('// generated using cog:')
        cog.outl('cout << "public api, shouldnt change within major version:" << endl;')
        for option in options:
            name = option['name']
            description = option['description']
            if 'ispublicapi' in option and option['ispublicapi']:
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
        cog.outl('cout << "" << endl; ')
        cog.outl('cout << "unstable, might change within major version:" << endl; ')
        for option in options:
            if 'ispublicapi' not in option or not option['ispublicapi']:
                name = option['name']
                description = option['description']
                cog.outl('cout << "    ' + name.lower() + '=[' + description + '] (" << config.' + name + ' << ")" << endl;')
    *///]]]
    // generated using cog:
    cout << "public api, shouldnt change within major version:" << endl;
    cout << "    gpuindex=[gpu device index; default value is gpu if present, cpu otw.] (" << config.gpuIndex << ")" << endl;
    cout << "    weightsfile=[file to read weights from] (" << config.weightsFile << ")" << endl;
    cout << "    batchsize=[batch size] (" << config.batchSize << ")" << endl;
    cout << "" << endl; 
    cout << "unstable, might change within major version:" << endl; 
    cout << "    inputfile=[file to read inputs from, if empty, read stdin (default)] (" << config.inputFile << ")" << endl;
    cout << "    outputfile=[file to write outputs to, if empty, write to stdout] (" << config.outputFile << ")" << endl;
    cout << "    outputlayer=[layer to write output from, default -1 means: last layer] (" << config.outputLayer << ")" << endl;
    cout << "    writelabels=[write integer labels, instead of probabilities etc (default 0)] (" << config.writeLabels << ")" << endl;
    cout << "    outputformat=[output format [binary|text]] (" << config.outputFormat << ")" << endl;
    // [[[end]]]
}

int PredictionModel::predictCmd(std::string argument){

    istringstream iss(argument);
    vector<string> tokens;
    //LOGI ("1");
    copy(istream_iterator<string>(iss),
         istream_iterator<string>(),
         back_inserter(tokens));
    //LOGI ("2");
    for(int i =0;i<tokens.size();i++){
    	//LOGI ("%s\n",tokens[i].c_str());
    }
    //LOGI ("3");
    char** argList = new char*[tokens.size()];
    for(int i = 0; i < tokens.size(); ++i)
    {
    	argList[i] = new char[tokens[i].length()+1];
    	memcpy ( argList[i], tokens[i].c_str(),  tokens[i].length() );
    	argList[i][tokens[i].length()]='\0';
    }
	//for(int j=0;j<tokens.size();j++){
		  //LOGI ("%s\n",argList[j]);

	//  }

	int i=prepareConfig(tokens.size(), argList);

	tokens.clear();
	//LOGI ("finish2");
    for (int j=0; j<tokens.size(); j++)
    	delete argList[j];



    delete[] argList;


	return 1;
}

int PredictionModel::prepareConfig(int parameterNb, char *argList[]) {
    ConfigPrediction config;
    if(parameterNb == 2 && (string(argList[1]) == "--help" || string(argList[1]) == "--?" || string(argList[1]) == "-?" || string(argList[1]) == "-h") ) {
        printUsage(argList, config);
        //LOGE( "printUsage");
    }
    //LOGE( "debut");
    for(int i = 1; i < parameterNb; i++) {
        vector<string> splitkeyval = split(argList[i], "=");
        if(splitkeyval.size() != 2) {
          cout << "Usage: " << argList[0] << " [key]=[value] [[key]=[value]] ..." << endl;
          exit(1);
        } else {
            string key = splitkeyval[0];
            string value = splitkeyval[1];
            //LOGE( "key=%s value=%s ",key.c_str(),value.c_str());
//            cout << "key [" << key << "]" << endl;
            /* [[[cog
                cog.outl('// generated using cog:')
                cog.outl('if(false) {')
                for option in options:
                    name = option['name']
                    type = option['type']
                    cog.outl('} else if(key == "' + name.lower() + '") {')
                    converter = '';
                    if type == 'int':
                        converter = 'atoi';
                    elif type == 'float':
                        converter = 'atof';
                    cog.outl('    config.' + name + ' = ' + converter + '(value);')
            */// ]]]
            // generated using cog:
            if(false) {
            } else if(key == "gpuindex") {
                config.gpuIndex = atoi(value);
            } else if(key == "weightsfile") {
                config.weightsFile = (value);
            } else if(key == "batchsize") {
                config.batchSize = atoi(value);
            } else if(key == "inputfile") {
                config.inputFile = (value);
            } else if(key == "outputfile") {
                config.outputFile = (value);
            } else if(key == "outputlayer") {
                config.outputLayer = atoi(value);
            } else if(key == "writelabels") {
                config.writeLabels = atoi(value);
            } else if(key == "outputformat") {
                config.outputFormat = (value);
            // [[[end]]]
            } else {
            	//LOGE( "Error: key %s not recognised",key.c_str());
//                cout << endl;
//                cout << "Error: key '" << key << "' not recognised" << endl;
//                cout << endl;
//                printUsage(argList, config);
//                cout << endl;
                return -1;
            }
        }
    }
    if(config.outputFormat != "text" && config.outputFormat != "binary") {
        cout << endl;
        cout << "outputformat must be 'text' or 'binary'" << endl;
        cout << endl;
        return -1;
    }

    WormUpGPU(config);//the workload is very low (the result might be wrong if we don't worm up the gpu)

    try {
        go(config);
    } catch(runtime_error e) {
        cout << "Something went wrong: " << e.what() << endl;
        return -1;
    }
}

int PredictionModel::WormUpGPU(ConfigPrediction config) {
//	//the workload is very low (the result might be wrong if we don't worm up the gpu)
//	LOGI("worm up");
//    int N = -1;
//    int numPlanes;
//    int imageSize;
//    int imageSizeCheck;
//    GenericLoaderv2* loader = NULL;
//    if(config.inputFile == "") {
//        int dims[3];
//        cin.read(reinterpret_cast< char * >(dims), 3 * 4l);
//        numPlanes = dims[0];
//        imageSize = dims[1];
//        imageSizeCheck = dims[2];
//        if(imageSize != imageSizeCheck) {
//        	LOGE("imageSize doesnt match imageSizeCheck, image not square");
//            throw std::runtime_error("imageSize doesnt match imageSizeCheck, image not square");
//        }
//    } else {
//        loader = new GenericLoaderv2(config.inputFile);
//        N = loader->getN();
//        numPlanes = loader->getPlanes();
//        imageSize = loader->getImageSize();
//
//    }
//
//    const long inputCubeSize = numPlanes * imageSize * imageSize ;
//
//    if ((inputCubeSize*config.batchSize)<(32*1024) ){//check if weneed to worm up the gpu
//
//		NeuralNet *net;
//		net = new NeuralNet(cl);
//
//		WeightsInitializer *weightsInitializer = new OriginalInitializer();
//
//		config.batchSize=N;
//		string netDef("default netDef");
//		WeightsPersister::loadConfigString(config.weightsFile, netDef);
//
//		net->addLayer(InputLayerMaker::instance()->numPlanes(numPlanes)->imageSize(imageSize));
//
//		net->addLayer(NormalizationLayerMaker::instance()->translate(0.0)->scale(1.0) ); // This will be read from weights file
//
//		NetdefToNet::createNetFromNetdef(config.batchSize,net, netDef, weightsInitializer);
//
//		int ignI;
//			float ignF;
//
//			WeightsPersister::loadWeights(config.weightsFile, string("netDef=")+netDef, net, &ignI, &ignI, &ignF, &ignI, &ignF,false);
//			net->setBatchSize(config.batchSize);
//
//			float *inputData = new float[ inputCubeSize * config.batchSize];
//
//				int *labels = new int[config.batchSize];
//				int n = 0;
//				bool more = true;
//				ostream *outFile = 0;
//				if(config.outputFile == "") {
//					outFile = &cout;
//				} else {
//					if(config.outputFormat == "text") {
//						outFile = new ofstream(config.outputFile, ios::out);
//					} else if(config.outputFormat == "binary") {
//						outFile = new ofstream(config.outputFile, ios::out | std::ios::binary);
//					} else {
//						throw runtime_error("outputFormat " + config.outputFormat + " not recognized");
//					}
//				}
//				if(config.outputLayer == -1) {
//					config.outputLayer = net->getNumLayers() - 1;
//				}
//				if(config.inputFile == "") {
//					cin.read(reinterpret_cast< char * >(inputData), inputCubeSize * config.batchSize * 4l);
//					more = !cin.eof();
//				} else {
//
//					loader->load(inputData, 0, n, config.batchSize);
//				}
//
//
//
//		for(int i =0; i<512;i++){
//
//			dynamic_cast<InputLayer *>(net->getLayer(0))->in(inputData);
//
//			for(int layerId = 0; layerId <= config.outputLayer; layerId++) {
//				net->getLayer(layerId)->forward();
//			}
//
//		}
//
//		if(loader != NULL) delete loader;
//		delete[] inputData;
//		delete[] labels;
//		delete weightsInitializer;
//		delete net;
//    }
}



