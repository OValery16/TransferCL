// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include <iostream>
#include <stdexcept>
#include <cstring>

#include "../../EasyCL/EasyCL.h"
#include "PoolingBackward.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/stringhelper.h"

#include "PoolingBackwardGpuNaive.h"

#define MEASURE_BACKWARD_PROP 0

using namespace std;

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

VIRTUAL PoolingBackwardGpuNaive::~PoolingBackwardGpuNaive() {
#if TRANSFERCL_VERBOSE == 1
	LOGI( "DeepCL/src/pooling/PoolingBackwardGpuNaive.cpp: ~PoolingBackwardGpuNaive");
#endif

	if (test){
		delete kernel;
		delete kMemset;
	}

    delete kernel2;
}
VIRTUAL void PoolingBackwardGpuNaive::backward(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *selectorsWrapper, 
        CLWrapper *gradInputWrapper,CLWrapper * inputWrapper){
	#if MEASURE_BACKWARD_PROP == 1
		LOGE("backward pooling");
		//inputWrapper->copyToDevice();// problem it seems th buffer is durty. Check why ?
		LOGI("isOnDevice %d",inputWrapper->isDeviceDirty());
	#endif
    kernel2->in(batchSize)->inout(gradOutputWrapper)->in(selectorsWrapper)->in(gradInputWrapper)->in(inputWrapper);
    if (not setup) {
		workgroupSize2 = kernel2->get_kernel_work_group_size();;
		numWorkgroups2 = (batchSize * numPlanes * outputSize * outputSize + workgroupSize2 - 1) / workgroupSize2;
		setup= true;
	}
	#if MEASURE_BACKWARD_PROP == 1
		LOGI("globalSize %d workgroupsize %d",numWorkgroups2 * workgroupSize2,workgroupSize2);
	#endif
    kernel2->run_1d(numWorkgroups2 * workgroupSize2, workgroupSize2);
    cl->finish();
	#if MEASURE_BACKWARD_PROP == 1
		if (test){
			inputWrapper->copyToHost();
			float * grad0=(float*)inputWrapper->getHostArray();
			gradInputWrapper->copyToHost();
			float* grad = (float*)gradInputWrapper->getHostArray();

			float* temp=new float[batchSize * numPlanes * inputSize * inputSize];

			for (int i=0; i<batchSize * numPlanes * inputSize * inputSize;i++)
				temp[i]=grad[i];

			////////////////////////////////

			gradOutputWrapper->copyToHost();
			selectorsWrapper->copyToHost();
			int * selectors=(int*)selectorsWrapper->getHostArray();
			float * gradOutTest=(float *)gradOutputWrapper->getHostArray();
			float *gradInputTest = new float[ getInputNumElements(batchSize) ];
			memset(gradInputTest, 0, sizeof(float) * getInputNumElements(batchSize) );
				for(int n = 0; n < batchSize; n++) {
					for(int plane = 0; plane < numPlanes; plane++) {
						for(int outputRow = 0; outputRow < outputSize; outputRow++) {
							int inputRow = outputRow * poolingSize;
							for(int outputCol = 0; outputCol < outputSize; outputCol++) {
								int inputCol = outputCol * poolingSize;
								int outputIndex = getResultIndex(n, plane, outputRow, outputCol);
								int selector = selectors[outputIndex];
								int drow = selector / poolingSize;
								int dcol = selector % poolingSize;
								int inputIndex = getInputIndex(n, plane, inputRow + drow, inputCol + dcol);
								gradInputTest[ inputIndex ] = (grad0[outputIndex]>0 ? grad0[outputIndex]:0.0f)*gradOutTest[outputIndex];
							}
						}
					}
				}

	//olivier error with the code
	//		StatefulTimer::instance()->timeCheck("PoolingBackwardGpuNaive::backward start");
	//
	//		// first, memset errors to 0 ...
	//		kMemset->out(gradInputWrapper)->in(0.0f)->in(batchSize * numPlanes * inputSize * inputSize);
	//		int globalSize = batchSize * numPlanes * inputSize * inputSize;
	//		int workgroupSize = 64;
	//		int numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
	//		kMemset->run_1d(numWorkgroups * workgroupSize, workgroupSize);
	//		cl->finish();
	//		LOGE("backward pooling1");
	//		kernel->in(batchSize)->inout(gradOutputWrapper)->in(selectorsWrapper)->in(gradInputWrapper);
	//		globalSize = batchSize * numPlanes * outputSize * outputSize;
	//		workgroupSize = 64;
	//		numWorkgroups = (globalSize + workgroupSize - 1) / workgroupSize;
	//		kernel->run_1d(numWorkgroups * workgroupSize, workgroupSize);
	//		cl->finish();
	//		LOGE("backward pooling2");
	//		StatefulTimer::instance()->timeCheck("PoolingBackwardGpuNaive::backward end");
	//
	//		gradInputWrapper->copyToHost();
	//		grad = (float*)gradInputWrapper->getHostArray();


			//inputWrapper->copyToHost();
			//float* input = (float*)inputWrapper->getHostArray();

			float error=0.0f;
			for (int i=0; i<batchSize * numPlanes * inputSize * inputSize;i++)
				error+=abs(temp[i]-gradInputTest[i]);

				for (int i=0; i<10;i++)
					LOGI("temp[%d]=%f other[%d]=%f",i,temp[i],i,gradInputTest[i]);


			LOGI("error pooling %f", error);
			delete[] temp;
			delete[] gradInputTest;
		}
	#endif

}
PoolingBackwardGpuNaive::PoolingBackwardGpuNaive(EasyCL*cl, bool padZeros, int numPlanes, int inputSize, int poolingSize,int previousLayer_activationLayer,bool bool_test) :
        PoolingBackward(cl, padZeros, numPlanes, inputSize, poolingSize) {

	test=bool_test;//1;//
	setup=false;

	kernel=0;
	kernel2=0;

	if (test){
		string options = "";
		options += " -D gNumPlanes=" + toString(numPlanes);
		options += " -D gInputSize=" + toString(inputSize);
		options += " -D gInputSizeSquared=" + toString(inputSize * inputSize);
		options += " -D gOutputSize=" + toString(outputSize);
		options += " -D gOutputSizeSquared=" + toString(outputSize * outputSize);
		options += " -D gPoolingSize=" + toString(poolingSize);
		options += " -D gPadZeros=" + toString(padZeros ? 1 : 0);

		// [[[cog
		// import stringify
		// stringify.write_kernel2("kernel", "cl/PoolingBackwardGpuNaive.cl", "backward", 'options')
		// stringify.write_kernel2("kMemset", "cl/memset.cl", "cl_memset", '""')
		// ]]]
		// generated using cog, from cl/PoolingBackwardGpuNaive.cl:
		const char * kernelSource =
		"// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
		"//\n"
		"// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
		"// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
		"// obtain one at http://mozilla.org/MPL/2.0/.\n"
		"\n"
		"// inplane and outplane are always identical, 1:1 mapping, so can just write `plane`\n"
		"// gradOutput: [n][plane][outrow][outcol]\n"
		"// selectors: [n][plane][outrow][outcol]\n"
		"// gradInput: [n][plane][inrow][incol]\n"
		"// wont use workgroups (since 'naive')\n"
		"// one thread per: [n][plane][outrow][outcol]\n"
		"// globalId: [n][plane][outrow][outcol]\n"
		"kernel void backward(const int batchSize,\n"
		"    global const float *gradOutput, global const int *selectors, global float *gradInput) {\n"
		"\n"
		"    #define globalId get_global_id(0)\n"
		"    #define nPlaneCombo (globalId / gOutputSizeSquared)\n"
		"    #define outputPosCombo (globalId % gOutputSizeSquared)\n"
		"\n"
		"    const int n = nPlaneCombo / gNumPlanes;\n"
		"    const int plane = nPlaneCombo % gNumPlanes;\n"
		"    const int outputRow = outputPosCombo / gOutputSize;\n"
		"    const int outputCol = outputPosCombo % gOutputSize;\n"
		"\n"
		"    if (n >= batchSize) {\n"
		"        return;\n"
		"    }\n"
		"\n"
		"    int resultIndex = (( n\n"
		"        * gNumPlanes + plane)\n"
		"        * gOutputSize + outputRow)\n"
		"        * gOutputSize + outputCol;\n"
		"    #define error (gradOutput[resultIndex])\n"
		"    int selector = (selectors[resultIndex]);\n"
		"    #define drow (selector / gPoolingSize)\n"
		"    #define dcol (selector % gPoolingSize)\n"
		"    #define inputRow (outputRow * gPoolingSize + drow)\n"
		"    #define inputCol (outputCol * gPoolingSize + dcol)\n"
		"    int inputIndex = (( n\n"
		"        * gNumPlanes + plane)\n"
		"        * gInputSize + inputRow)\n"
		"        * gInputSize + inputCol;\n"
		"//    if (n < batchSize) {\n"
		"        gradInput[ inputIndex ] = error;\n"
		"//    }\n"
		"}\n"
		"\n"
		"";
		kernel = cl->buildKernelFromString("PoolingBackwardGpuNaive", kernelSource, "backward", options, "cl/PoolingBackwardGpuNaive.cl");
		// generated using cog, from cl/memset.cl:
		const char * kMemsetSource =
		"// Copyright Hugh Perkins 2015 hughperkins at gmail\n"
		"//\n"
		"// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
		"// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
		"// obtain one at http://mozilla.org/MPL/2.0/.\n"
		"\n"
		"kernel void cl_memset(global float *target, const float value, const int N) {\n"
		"    #define globalId get_global_id(0)\n"
		"    if ((int)globalId < N) {\n"
		"        target[globalId] = value;\n"
		"    }\n"
		"}\n"
		"\n"
		"";
		kMemset = cl->buildKernelFromString("PoolingBackwardGpuNaive", kMemsetSource, "cl_memset", "", "cl/memset.cl");
	}

    string kernelSource2 =    "kernel void backward_Pooling(const int batchSize,\n"
        "    const __global float* restrict gradOutput, const __global int* restrict selectors, global float *gradInput,const __global float* restrict inputs) {\n"
        "\n"
        "    int globalId=get_global_id(0);\n"
        "    int nPlaneCombo=(globalId / {{gOutputSizeSquared}});\n"
        "    int outputPosCombo=(globalId % {{gOutputSizeSquared}});\n"
        "\n"
        "    const int n = nPlaneCombo / {{gNumPlanes}};\n"
        "    const int plane = nPlaneCombo % {{gNumPlanes}};\n"
        "    const int outputRow = outputPosCombo / {{gOutputSize}};\n"
        "    const int outputCol = outputPosCombo % {{gOutputSize}};\n"
        "\n"
        "    int resultIndex = (( n* {{gNumPlanes}} + plane)* {{gOutputSize}} + outputRow)* {{gOutputSize}} + outputCol;\n"
        "    float error={{gActivationFunction}}\n"
        "    int selector = (selectors[resultIndex]);\n"
        "    int drow=(selector / {{gPoolingSize}});\n"
        "    int dcol=(selector % {{gPoolingSize}});\n"
    	"{{gPoolingBackprop}}"
        "}\n"
        "\n"
        "";

    buildKernelBackward(kernelSource2,previousLayer_activationLayer);
}

void PoolingBackwardGpuNaive::buildKernelBackward( string kernelSource,int previousLayer_activationLayer) {
    TemplatedKernel builder(cl);


        setupBuilderBackward(&builder,previousLayer_activationLayer);

        //string identifier2="backward_Pooling"+std::to_string(inputSize);

    	string identifier2="backward_Pooling";
    		 identifier2=identifier2+"poolingSize=";
    		 identifier2=identifier2+std::to_string(poolingSize);
    		 identifier2=identifier2+"_InputSize="+std::to_string(inputSize);
    		 //identifier2=identifier2+"_batchsize="+std::to_string(batchsize);
    		 identifier2=identifier2+"_OutputSize="+std::to_string(outputSize);
    		 identifier2=identifier2+"_gPadZeros="+BoolToString(padZeros);
    		 identifier2=identifier2+"_plane="+std::to_string(numPlanes);

        this->kernel2 = builder.buildKernel(
           		identifier2,
               "backward_Pooling",
               kernelSource.c_str(),
               "backward_Pooling",
               false
        );
    }

void PoolingBackwardGpuNaive::setupBuilderBackward(TemplatedKernel *builder,int previousLayer_activationLayer){

	builder->set("gInputSizeSquared",inputSize*inputSize);
	builder->set("gInputSize",inputSize);
	builder->set("gPadZeros",padZeros ? 1 : 0);
	builder->set("gNumPlanes",numPlanes);
	builder->set("gOutputSize",outputSize);
	builder->set("gOutputSizeSquared",outputSize*outputSize);
	builder->set("gPoolingSize",poolingSize);

	setActivationFunction(builder,previousLayer_activationLayer);

	string gPoolingBackpropString="";

	for(int i=0;i<poolingSize;i++)
		for(int j=0;j<poolingSize;j++)
			gPoolingBackpropString+="    gradInput[(( n* {{gNumPlanes}} + plane)* {{gInputSize}} + (outputRow * {{gPoolingSize}} +"+to_string(i)+"))* {{gInputSize}} + (outputCol * {{gPoolingSize}} + "+to_string(j)+") ] = error*select(0.0f,1.0f,((drow=="+to_string(i)+")&&(dcol=="+to_string(j)+")));\n";


	builder->set("gPoolingBackprop",gPoolingBackpropString);
}


void  PoolingBackwardGpuNaive::setActivationFunction(TemplatedKernel *builder,int previousLayer_activationLayer){

	string replaceString=" ";



		if (previousLayer_activationLayer==-1)//no activation layer
			replaceString= "(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==1)
			replaceString=  "(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==2)
			replaceString=  "fmax ( inputs[globalId] , 0 )*(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==3)
			replaceString=  "(1 - inputs[globalId] * inputs[globalId])*(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==4)
			replaceString=  "(0.66667f * (1.7159f - 1 / 1.7159f * inputs[globalId] * inputs[globalId]) )*(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==5)
			replaceString=  "(inputs[globalId] * (1 - inputs[globalId]) )*(gradOutput[resultIndex]);";
		if (previousLayer_activationLayer==6)
			replaceString=  "(select(inputs[globalId]+1,1,fmax ( inputs[globalId] , 0 )))*(gradOutput[resultIndex]);";


		builder->set("gActivationFunction", replaceString);
}




