// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
#include "../../EasyCL/util/StatefulTimer.h"

#include "BackwardGpuNaive.h"

using namespace std;

#define MEASURE_BACKWARD_PROP 0
#define TEST_KERNEL 0

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

VIRTUAL BackwardGpuNaive::~BackwardGpuNaive() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackwardGpuNaive.cpp: ~BackwardGpuNaive");
#endif

	delete kernel2;
#if TEST_KERNEL == 1
    if (dim.test)
    	delete kernel;
#endif

}
VIRTUAL void BackwardGpuNaive::backward(int batchSize, 
        CLWrapper *inputDataWrapper, CLWrapper *gradOutputWrapper, CLWrapper *weightsWrapper,
        CLWrapper *gradInputWrapper) {

#if MEASURE_BACKWARD_PROP
    LOGI("BackwardGpuNaive");
	float *grad=0;
	float * temp = 0;
	float * temp2 = 0;

    if (dim.test){
			inputDataWrapper->copyToHost();
			float *input=(float*)inputDataWrapper->getHostArray();
			for(int i= 0; i< 10; i++)
				LOGI("input %f",input[i]);

			bool isZero=true;
			for(int i= 0; i< batchSize * dim.inputCubeSize; i++)
				if (input[i]!=0)
					isZero=false;
			LOGI("isZero %d",isZero);
			kernel
			   ->in(batchSize)
				->in(gradOutputWrapper)
			   ->in(weightsWrapper)
				->out(gradInputWrapper);


			globalSize = batchSize * dim.inputCubeSize;
			workgroupsize = kernel->get_kernel_work_group_size();//cl->getMaxWorkgroupSize();
			globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
			LOGI("globalSize %d workgroupsize %d",globalSize,workgroupsize);
			kernel->run_1d(globalSize, workgroupsize);
			cl->finish();

			gradInputWrapper->copyToHost();
			grad=(float*)gradInputWrapper->getHostArray();
			temp = new float[batchSize * dim.inputCubeSize];
			temp2 = new float[batchSize * dim.inputCubeSize];

			for(int i= 0; i< batchSize * dim.inputCubeSize; i++){
				temp[i]=grad[i];
			}

			if (dim.previousLayer_activationLayer==-1){
				for(int i= 0; i< batchSize * dim.inputCubeSize; i++){
						temp2[i]=grad[i];
					}
			}
			if (dim.previousLayer_activationLayer==1){
				for(int i= 0; i< batchSize * dim.inputCubeSize; i++){
						temp2[i]=grad[i];
					}
			}

			if (dim.previousLayer_activationLayer==3){
				for(int i= 0; i< batchSize * dim.inputCubeSize; i++){
						temp2[i]=(1 - input[i] * input[i])*grad[i];//(0.66667f * (1.7159f - 1 / 1.7159f * input[i] * input[i]) ) * grad[i];
					}
			}
    }
#endif

if ( not setup){
    globalSize = batchSize * dim.inputCubeSize;
    workgroupsize = kernel2->get_kernel_work_group_size();
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
    setup=true;

    kernel2
          ->in(batchSize)
           ->in(gradOutputWrapper)
          ->in(weightsWrapper)
           ->out(gradInputWrapper)
           ->in(inputDataWrapper);
}

    kernel2->run_1d(globalSize, workgroupsize);
    //cl->finish();
#if MEASURE_BACKWARD_PROP
    if (dim.test){
		gradInputWrapper->copyToHost();
		float*grad2=(float*)gradInputWrapper->getHostArray();
		float error=0.0f;
		for(int i= 0; i< batchSize * dim.inputCubeSize; i++)
			error= abs(temp2[i]-grad2[i]);
		LOGI("conv) error backprop %f",error);

		for(int i= 0; i< 10; i++)
			LOGI("conv) error backprop %f %f",temp2[i],grad2[i]);


		gradInputWrapper->copyToDevice();
		delete[] temp;
		delete[] temp2;
    }
#endif
}

void BackwardGpuNaive::buildKernelBackward( string kernelSource) {
    TemplatedKernel builder(cl);

    //string identifier2="BackpropNaive2"+std::to_string(dim.numFilters);

	string identifier2="BackpropNaive2";
		 identifier2=identifier2+"nbFilter=";
		 identifier2=identifier2+std::to_string(dim.numFilters);
		 identifier2=identifier2+"_InputSize="+std::to_string(dim.inputSize);
		 identifier2=identifier2+"_batchsize="+std::to_string(dim.batchsize);
		 identifier2=identifier2+"_OutputSize="+std::to_string(dim.outputSize);
		 identifier2=identifier2+"_conv="+BoolToString(dim.isConv);
		 identifier2=identifier2+"_normalize="+BoolToString(dim.needToNormalize);
		 identifier2=identifier2+"_maxpool="+BoolToString(dim.useMaxPooling);


    	ConfigManager*binariesManager=new ConfigManager( cl->absolutePath/*path,binaryFileRep*/);
    	bool compiled;
    	string filepath="default";
    	if (not binariesManager->alreadyCompiledKernel("calcGradInput","",identifier2,filepath)){

    		inferenceBackward(kernelSource);
    		setActivationFunction(&builder);
    		setupBuilderBackward(&builder);
    	}

    this->kernel2 = builder.buildKernel(
           		identifier2,
               "BackpropNaive",
               kernelSource.c_str(),
               "calcGradInput",
               false
        );


    }

void BackwardGpuNaive::inferenceBackward(string &kernelSource) {

	string kernelSource2="void kernel calcGradInput(\n"
		    "        const int batchSize,\n"
		    "        const __global float* restrict gradOutput, global float *weights, global float *gradInput,const __global float* restrict inputs) {\n"
		    "    int globalId = get_global_id(0);\n"
		    "\n"
		    "    const int upstreamImage2dId = globalId ;\n"
		    "\n"
		    "\n"
		    "    const int upstreamPlane = upstreamImage2dId % {{gInputPlanes}};\n"
		    "    const int n = upstreamImage2dId / {{gInputPlanes}};\n"
		    "\n"
		    "    const int minFilterRow = max(0, 0 + {{gMargin}} );\n"
		    "    const int maxFilterRow = min({{gFilterSize}} - 1, 0 + {{gMargin}});\n"
		    "    const int minFilterCol = max(0, 0 + {{gMargin}} );\n"
		    "    const int maxFilterCol = min({{gFilterSize}} - 1, 0 + {{gMargin}});\n"
		    "\n"
		    "    float sumWeightTimesOutError = 0;\n"
		    "    // aggregate over [outPlane][outRow][outCol]\n"
			"    #pragma unroll\n"
		    "    for (int outPlane = 0; outPlane < {{gNumFilters}}; outPlane++) {\n"
			"        #pragma unroll\n"
		    "        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
		    "            int outRow =  {{gMargin}} - filterRow;\n"
			"            #pragma unroll\n"
		    "            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
		    "                int outCol =  {{gMargin}} - filterCol;\n"
		    "                int resultIndex = (( n * {{gNumFilters}}\n"
		    "                          + outPlane)\n"
		    "                          + outRow)\n"
		    "                          + outCol;\n"
		    "                float thisError = gradOutput[resultIndex];\n"
		    "                int thisWeightIndex = (( outPlane * {{gInputPlanes}}\n"
		    "                                    + upstreamPlane) * {{gFilterSize}}\n"
		    "                                    + filterRow) * {{gFilterSize}}\n"
		    "                                    + filterCol;\n"
		    "                float thisWeight = weights[thisWeightIndex];\n"
		    "                float thisWeightTimesError = thisWeight * thisError;\n"
		    "                sumWeightTimesOutError += thisWeightTimesError;\n"
		    "            }\n"
		    "        }\n"
		    "    }\n"
		    "    gradInput[globalId] = {{gActivationFunction}};\n"
		    "}\n"
		    "\n"
		    "";

	string kernelSource3="void kernel calcGradInput(\n"
		    "        const int batchSize,\n"
		    "        const __global float* restrict gradOutput, global float *weights, global float *gradInput,const __global float* restrict inputs) {\n"
			    "    int globalId = get_global_id(0);\n"
			    "\n"
			    "    const int upstreamImage2dId = globalId ;\n"
			    "\n"
			    "\n"
			    "    const int upstreamPlane = upstreamImage2dId % {{gInputPlanes}};\n"
			    "    const int n = upstreamImage2dId / {{gInputPlanes}};\n"
			    "\n"
			    "    const int minFilterRow = 0;\n"
			    "    const int maxFilterRow = 0;\n"
			    "    const int minFilterCol = 0;\n"
			    "    const int maxFilterCol = 0;\n"
			    "\n"
			    "    float sumWeightTimesOutError = 0;\n"
			    "    // aggregate over [outPlane][outRow][outCol]\n"
				"    #pragma unroll\n"
			    "    for (int outPlane = 0; outPlane < {{gNumFilters}}; outPlane++) {\n"
			    "                float thisError = gradOutput[( n * {{gNumFilters}}+ outPlane)];\n"
			    "                float thisWeight = weights[(( outPlane * {{gInputPlanes}}+ upstreamPlane))];\n"
			    "                sumWeightTimesOutError += thisWeight * thisError;\n"
			    "    }\n"
			    "    gradInput[globalId] = {{gActivationFunction}};\n"// (0.66667f * (1.7159f - 1 / 1.7159f * inputs[globalId] * inputs[globalId]) ) * sumWeightTimesOutError;
			    "}\n"
			    "\n"
			    "";


	int margin=dim.padZeros ? dim.filterSize >> 1 : 0;


	if ((dim.inputSize==1)&&(dim.outputSize==1)){
		if ((margin==0)&&(dim.filterSize==1))
			kernelSource=kernelSource3;
		else
			kernelSource=kernelSource2;
	}

}

void  BackwardGpuNaive::setActivationFunction(TemplatedKernel *builder){
//	if (_activ == "linear")
//		this->activationLayer=1;
//	if (_activ=="relu")
//		this->activationLayer=2;
//	if (_activ=="tanh")
//		this->activationLayer=3;
//	if (_activ=="scaledtanh")
//		this->activationLayer=4;
//	if (_activ=="sigmoid")
//		this->activationLayer=5;
//	if (_activ=="elu")
//		this->activationLayer=6;
//    "#ifdef TANH\n"
//    "    #define ACTIVATION_DERIV(output) (1 - output * output)\n"
//    "#elif defined SCALEDTANH\n"
//    "    #define ACTIVATION_DERIV(output) (0.66667f * (1.7159f - 1 / 1.7159f * output * output) )\n"
//    "#elif defined SIGMOID\n"
//    "    #define ACTIVATION_DERIV(output) (output * (1 - output) )\n"
//    "#elif defined RELU\n"
//    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : 0)\n"
//    "#elif defined ELU\n"
//    "    #define ACTIVATION_DERIV(output) (output > 0 ? 1 : output + 1)\n"
//    "#elif defined LINEAR\n"
//    "    #define ACTIVATION_DERIV(output) (1.0f)\n"


	string replaceString=" ";

		if (dim.previousLayer_activationLayer==-1)//no activation layer
			replaceString= "sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==1)
			replaceString=  "sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==2)
			replaceString=  "fmax ( inputs[globalId] , 0 )*sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==3)
			replaceString=  "(1 - inputs[globalId] * inputs[globalId])*sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==4)
			replaceString=  "(0.66667f * (1.7159f - 1 / 1.7159f * inputs[globalId] * inputs[globalId]) )*sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==5)
			replaceString=  "(inputs[globalId] * (1 - inputs[globalId]) )*sumWeightTimesOutError";
		if (dim.previousLayer_activationLayer==6)
			replaceString=  "(select(inputs[globalId]+1,1,fmax ( inputs[globalId] , 0 )))*sumWeightTimesOutError";

		//LOGI("3)dim.activationLayer %d %s",dim.activationLayer,replaceString.c_str());
		builder->set("gActivationFunction", replaceString);
}


void BackwardGpuNaive::setupBuilderBackward(TemplatedKernel *builder) {



	builder->set("gInputSizeSquared",square(dim.inputSize));
	builder->set("gInputSize",dim.inputSize);
	builder->set("gInputPlanes",dim.inputPlanes);
	builder->set("gMargin",dim.padZeros ? dim.filterSize >> 1 : 0);
	builder->set("gOutputSize",dim.outputSize);
	builder->set("gFilterSize",dim.filterSize);
	builder->set("gNumFilters",dim.numFilters);
	//setActivationFunction(builder);
}


BackwardGpuNaive::BackwardGpuNaive(EasyCL *cl, LayerDimensions dim) {
	this->cl=cl;
	this->dim=dim;
	setup= false;
    std::string options = dim.buildOptionsString();
    options += ""; // " -D " + upstreamFn->getDefineName();

    string kernelSource2 =
    		    "void kernel calcGradInput(\n"
    			"        const int batchSize,\n"
    			"        const __global float* restrict gradOutput, global float *weights, global float *gradInput,const __global float* restrict inputs) {\n"
    		    "    int globalId = get_global_id(0);\n"
    		    "\n"
    		    "    const int upstreamImage2dId = globalId / {{gInputSizeSquared}};\n"
    		    "\n"
    		    "    const int intraImageOffset = globalId % {{gInputSizeSquared}};\n"
    		    "    const int upstreamRow = intraImageOffset / {{gInputSize}};\n"
    		    "    const int upstreamCol = intraImageOffset % {{gInputSize}};\n"
    		    "\n"
    		    "    const int upstreamPlane = upstreamImage2dId % {{gInputPlanes}};\n"
    		    "    const int n = upstreamImage2dId / {{gInputPlanes}};\n"
    		    "\n"
    		    "    if (n >= batchSize) {\n"
    		    "        return;\n"
    		    "    }\n"
    		    "\n"
    		    "    const int minFilterRow = max(0, upstreamRow + {{gMargin}} - ({{gOutputSize}} - 1));\n"
    		    "    const int maxFilterRow = min({{gFilterSize}} - 1, upstreamRow + {{gMargin}});\n"
    		    "    const int minFilterCol = max(0, upstreamCol + {{gMargin}} - ({{gOutputSize}} -1));\n"
    		    "    const int maxFilterCol = min({{gFilterSize}} - 1, upstreamCol + {{gMargin}});\n"
    		    "\n"
    		    "    float sumWeightTimesOutError = 0;\n"
    		    "    // aggregate over [outPlane][outRow][outCol]\n"
    		    "    #pragma unroll\n"
    		    "    for (int outPlane = 0; outPlane < {{gNumFilters}}; outPlane++) {\n"
    		    "        #pragma unroll\n"
    		    "        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
    		    "            int outRow = upstreamRow + {{gMargin}} - filterRow;\n"
    		    "            #pragma unroll\n"
    		    "            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
    		    "                int outCol = upstreamCol + {{gMargin}} - filterCol;\n"
    		    "                int resultIndex = (( n * {{gNumFilters}}\n"
    		    "                          + outPlane) * {{gOutputSize}}\n"
    		    "                          + outRow) * {{gOutputSize}}\n"
    		    "                          + outCol;\n"
    		    "                float thisError = gradOutput[resultIndex];\n"
    		    "                int thisWeightIndex = (( outPlane * {{gInputPlanes}}\n"
    		    "                                    + upstreamPlane) * {{gFilterSize}}\n"
    		    "                                    + filterRow) * {{gFilterSize}}\n"
    		    "                                    + filterCol;\n"
    		    "                float thisWeight = weights[thisWeightIndex];\n"
    		    "                float thisWeightTimesError = thisWeight * thisError;\n"
    		    "                sumWeightTimesOutError += thisWeightTimesError;\n"
    		    "            }\n"
    		    "        }\n"
    		    "    }\n"
    		    "    gradInput[globalId] = {{gActivationFunction}};\n"
    		    "}\n"
    		    "\n"
    		    "";
#if TEST_KERNEL == 1
    if (dim.test){
    	 const char * kernelSource =
    	    "// Copyright Hugh Perkins 2014 hughperkins at gmail\n"
    	    "//\n"
    	    "// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
    	    "// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
    	    "// obtain one at http://mozilla.org/MPL/2.0/.\n"
    	    "\n"
    	    "// expected defines:\n"
    	    "//  - none\n"
    	    "\n"
    	    "// globalid as: [n][upstreamPlane][upstreamrow][upstreamcol]\n"
    	    "// inputdata: [n][upstreamPlane][upstreamrow][upstreamcol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    	    "// gradOutput: [n][outPlane][outRow][outCol] 128 * 32 * 19 * 19 * 4 = 6MB\n"
    	    "// weights: [filterId][inputPlane][filterRow][filterCol] 32 * 32 * 5 * 5 * 4 = 409KB\n"
    	    "void kernel calcGradInput(\n"
    	    "        const int batchSize,\n"
    	    "        global const float *gradOutput, global float *weights, global float *gradInput) {\n"
    	    "    int globalId = get_global_id(0);\n"
    	    "\n"
    	    "    const int upstreamImage2dId = globalId / gInputSizeSquared;\n"
    	    "\n"
    	    "    const int intraImageOffset = globalId % gInputSizeSquared;\n"
    	    "    const int upstreamRow = intraImageOffset / gInputSize;\n"
    	    "    const int upstreamCol = intraImageOffset % gInputSize;\n"
    	    "\n"
    	    "    const int upstreamPlane = upstreamImage2dId % gInputPlanes;\n"
    	    "    const int n = upstreamImage2dId / gInputPlanes;\n"
    	    "\n"
    	    "    if (n >= batchSize) {\n"
    	    "        return;\n"
    	    "    }\n"
    	    "\n"
    	    "    const int minFilterRow = max(0, upstreamRow + gMargin - (gOutputSize - 1));\n"
    	    "    const int maxFilterRow = min(gFilterSize - 1, upstreamRow + gMargin);\n"
    	    "    const int minFilterCol = max(0, upstreamCol + gMargin - (gOutputSize -1));\n"
    	    "    const int maxFilterCol = min(gFilterSize - 1, upstreamCol + gMargin);\n"
    	    "\n"
    	    "    float sumWeightTimesOutError = 0;\n"
    	    "    // aggregate over [outPlane][outRow][outCol]\n"
    	    "    for (int outPlane = 0; outPlane < gNumFilters; outPlane++) {\n"
    	    "        for (int filterRow = minFilterRow; filterRow <= maxFilterRow; filterRow++) {\n"
    	    "            int outRow = upstreamRow + gMargin - filterRow;\n"
    	    "            for (int filterCol = minFilterCol; filterCol <= maxFilterCol; filterCol++) {\n"
    	    "                int outCol = upstreamCol + gMargin - filterCol;\n"
    	    "                int resultIndex = (( n * gNumFilters\n"
    	    "                          + outPlane) * gOutputSize\n"
    	    "                          + outRow) * gOutputSize\n"
    	    "                          + outCol;\n"
    	    "                float thisError = gradOutput[resultIndex];\n"
    	    "                int thisWeightIndex = (( outPlane * gInputPlanes\n"
    	    "                                    + upstreamPlane) * gFilterSize\n"
    	    "                                    + filterRow) * gFilterSize\n"
    	    "                                    + filterCol;\n"
    	    "                float thisWeight = weights[thisWeightIndex];\n"
    	    "                float thisWeightTimesError = thisWeight * thisError;\n"
    	    "                sumWeightTimesOutError += thisWeightTimesError;\n"
    	    "            }\n"
    	    "        }\n"
    	    "    }\n"
    	    "    gradInput[globalId] = sumWeightTimesOutError;\n"
    	    "}\n"
    	    "\n"
    	    "";

    	 string operation="BackpropWeightsNaive"+std::to_string(dim.numFilters);
		kernel = cl->buildKernelFromString(operation, kernelSource, "calcGradInput", options, "../../cl/backward.cl");
    }
#endif

    buildKernelBackward(kernelSource2);
}

