// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#include "Forward1.h"
#include "../util/stringhelper.h"
#include "../../EasyCL/util/StatefulTimer.h"

#if TEST_FORWARD==1
#include "AddBias.h"
#endif


#include <sstream>
#include <iomanip>


using namespace std;



#undef VIRTUAL
#undef STATIC
#define VIRTUAL
#define STATIC

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}

VIRTUAL Forward1::~Forward1() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/Forward1.cpp: ~Forward1");
#endif
#if TEST_FORWARD==1
	delete kernelH;
	delete kernel;
    delete addBias;
#endif
    delete test;
}

#if TEST_FORWARD==1
std::string to_string_with_precision2(const float a_value, const int n = 2)
{
	std::stringstream ss;
	if (a_value==0)
		ss << std::fixed << 0;
	else
		ss << std::fixed << std::setprecision(n) << a_value;
    return ss.str();
}


float *Forward1::convolv(int batchSize, float *inputData, float *weights) {

	float *output = new float[ dim.outputCubeSize * batchSize ];
    for(int n = 0; n < batchSize; n++) {
        for(int filter = 0; filter < dim.numFilters; filter++) {
            for(int outRow = 0; outRow < dim.outputSize; outRow += 1 + dim.skip) {
                for(int outCol = 0; outCol < dim.outputSize; outCol += 1 + dim.skip) {
                    float sum = 0;
                    for(int inPlane = 0; inPlane < dim.inputPlanes; inPlane++) {
//                        cout << "inplane=" << inPlane << endl;
                        for(int u = -dim.halfFilterSize; u <= dim.halfFilterSize; u++) {
                            int inRow = outRow * (dim.skip + 1) + u + (dim.padZeros ? 0 : dim.halfFilterSize);
//                                cout << "candidate inRow " << inRow << endl;
                            if(inRow < 0 || inRow > dim.inputSize - 1) {
                                continue;
                            }
                            int filterRow = u + dim.halfFilterSize;
                            for(int v = -dim.halfFilterSize; v <= dim.halfFilterSize; v++) {
                                int inCol = outCol * (dim.skip + 1) + v + (dim.padZeros ? 0 : dim.halfFilterSize);
                                int filterCol = v + dim.halfFilterSize;
                                if(inCol < 0 || inCol > dim.inputSize - 1) {
                                    continue;
                                }
                                int inputIndex = (( n
                                    * dim.inputPlanes + inPlane)
                                    * dim.inputSize + inRow)
                                    * dim.inputSize + inCol;
                                int weightIndex = (( filter
                                    * dim.inputPlanes + inPlane)
                                    * dim.filterSize  + filterRow)
                                    * dim.filterSize  + filterCol;
//                                    cout << "inpos " << inRow << "," << inCol << " outpos " << outRow << "," << outCol
//                                        << " filterpos " << filterRow << "," << filterCol << endl;
                                float sumchange = inputData[ inputIndex] * weights[ weightIndex ];
                                if(sumchange != 0) {
//                                        cout << inputData[inputIndex] << " * " << weights[weightIndex] << " = " << sumchange << endl;
                                }
                                sum += sumchange;
//                                cout << "inputIndex=" << inputIndex << " weightIndex=" << weightIndex <<
//                                    "  inputData[inputIndex]=" << inputData[inputIndex] << " weights[weightIndex]=" << weights[weightIndex] << " sumchange " << sumchange << " sum=" << sum << endl;
                            }
                        }
                    }

//                    sum = fn->calc(sum);
                    int outputIndex = (( n
                        * dim.numFilters + filter)
                        * dim.outputSize + outRow)
                        * dim.outputSize + outCol;
                    output[outputIndex] = sum;
//                    cout << "outputIndex=" << outputIndex << " sum=" << sum << " output[outputIndex]=" <<
//                        output[outputIndex] << endl;
                }
            }
        }
    }
    return output;
}
#endif

VIRTUAL void Forward1::forwardFloat(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper){
#if TEST_FORWARD==1

//	float * output=(float*)outputWrapper->getHostArray();
	clock_t startTimer1, stopTimer1;
	startTimer1=clock();

	setup=false;

	if (setup!=true){
		globalSize = batchSize * dim.outputCubeSize;
		workgroupsize = kernel->get_kernel_work_group_size();
		//int workgroupsize = std::min(globalSize, cl->getMaxWorkgroupSize());
		globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
	}

    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
    stopTimer1 = clock();
    LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

    startTimer1=clock();
    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("Forward1::forward after call forward");

stopTimer1 = clock();

	LOGI("convolution took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

	startTimer1=clock();
    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward1::forward END");
	stopTimer1 = clock();

	LOGI("bias took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

#endif
}


VIRTUAL void Forward1::forwardHalf(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper, CLWrapper *selectorWrapper, CLWrapper *gradInputWrapper){


#if TEST_FORWARD==1
	if (timeBenchmark)
		startTimer1=clock();
#endif

	float * selector =0;
	if (setup!=true){
		if (dim.useMaxPooling)
			globalSize = batchSize * dim.numFilters *dim. maxPool_sizeOutput*dim.maxPool_sizeOutput;
		else
			globalSize = batchSize * dim.outputCubeSize;
		workgroupsize = test->get_kernel_work_group_size();
		globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		setup = true;

		test->input(dataWrapper);
		test->input(weightsWrapper);
		test->output(outputWrapper);
		if(dim.biased)
			test->input(biasWrapper);

		if(normalization){
			float translate=dim.translate;
			float scale=dim.scale;
			test->input(translate);
			test->input(scale);

		}
		if (dim.useMaxPooling){
			test->output(selectorWrapper);
			test->output(gradInputWrapper);
		}
}



	#if TEST_FORWARD==1
		if (timeBenchmark){
			stopTimer1 = clock();
			LOGI("setup took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		}


    if (timeBenchmark)
		startTimer1=clock();
    #endif
    test->run_1d(globalSize, workgroupsize);
    //cl->finish();

	#if TEST_FORWARD==1
    	cl->finish();
		if (timeBenchmark){
			StatefulTimer::timeCheck("Forward1::forward after call forward");
			stopTimer1 = clock();
			LOGI("convolution test took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

			stopTimer1 = clock();

		}
	#endif


}



VIRTUAL void Forward1::forward(int batchSize, CLWrapper *dataWrapper, CLWrapper *weightsWrapper, CLWrapper *biasWrapper,
    CLWrapper *outputWrapper) {
#if TEST_FORWARD==1

	float * output=(float*)outputWrapper->getHostArray();

if (false/*dim.useHalfMemory*/){


	////////////////////////////
		biasWrapper->copyToHost();
		float *bias= (float *)biasWrapper->getHostArray();
			half *biasHalf=new half[batchSize * dim.outputCubeSize];
		for (int i=0; i<batchSize * dim.outputCubeSize;i++)
			biasHalf[i] = FloatToHalf(bias[i]);
		biasWrapper->copyToDevice();
		CLWrapper *biasHalfWrapper = (CLWrapper *)cl->wrap(batchSize * dim.outputCubeSize, biasHalf);
		biasHalfWrapper->createOnDevice();
		biasHalfWrapper->copyToDevice();
	/////
	dataWrapper->copyToHost();
	float *data= (float *)dataWrapper->getHostArray();
		half *dataHalf=new half[dim.inputCubeSize*batchSize];
		for (int i=0; i<dim.inputCubeSize*batchSize;i++)
			dataHalf[i] = FloatToHalf(data[i]);

		CLWrapper *dataHalfWrapper = (CLWrapper *)cl->wrap(dim.inputCubeSize*batchSize, dataHalf);
		dataHalfWrapper->createOnDevice();
		dataHalfWrapper->copyToDevice();

///////////
		float *weights= (float *)weightsWrapper->getHostArray();
		half *weightsHalf=new half[dim.inputPlanes*dim.filterSizeSquared*dim.numFilters];
		for (int j=0; j<dim.numFilters;j++)
			for (int i=0; i<dim.inputPlanes*dim.filterSizeSquared;i++)
				if (i<dim.inputPlanes*dim.filterSizeSquared)
					weightsHalf[j*dim.inputPlanes*dim.filterSizeSquared+i] = FloatToHalf(weights[dim.filterSizeSquared*dim.inputPlanes*j+i]);
				else
					weightsHalf[j*dim.inputPlanes*dim.filterSizeSquared+i] = FloatToHalf(0.0);
		CLWrapper *weightsHalfWrapper = (CLWrapper *)cl->wrap(/*dim.filterSizeSquared**/dim.inputPlanes*dim.filterSizeSquared*dim.numFilters/**dim.inputPlanes*/, weightsHalf);
		weightsHalfWrapper->createOnDevice();
		weightsHalfWrapper->copyToDevice();
///////////

	//////////////
	half *outputHalf=new half[batchSize * dim.outputCubeSize];
	for (int i=0; i<batchSize * dim.outputCubeSize;i++)
		outputHalf[i] = FloatToHalf(0.0f);
	CLWrapper *outputHalfWrapper= (CLWrapper *)cl->wrap(batchSize * dim.outputCubeSize,outputHalf);
	LOGI("3");
	outputHalfWrapper->createOnDevice();
	outputHalfWrapper->copyToDevice();
	//////////////

	clock_t startTimer2, stopTimer2;
	startTimer2=clock();
	StatefulTimer::timeCheck("Forward1::forward START");

	kernelH->in(batchSize);
	kernelH->input(dataHalfWrapper);
	kernelH->input(weightsHalfWrapper);
	kernelH->output(outputHalfWrapper);
	int globalSize2 = batchSize * dim.outputCubeSize;
	int workgroupsize2 = kernelH->get_kernel_work_group_size();
	LOGI("workgroupsize2 %d, workgroupsize2 %d",workgroupsize2, cl->getMaxWorkgroupSize());
	globalSize2 = (( globalSize2 + workgroupsize2 - 1) / workgroupsize2) * workgroupsize2;
	kernelH->run_1d(globalSize2, workgroupsize2);
	cl->finish();


	stopTimer2 = clock();
		double elapse2 = 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC;
		LOGI("convolution half took %g ms\n\n", 1000.0* (double)(stopTimer2 - startTimer2)/(double)CLOCKS_PER_SEC) ;

	if(dim.biased) {


	        addBias->forward2(
	            batchSize, dim.numFilters, dim.outputSize,
	            outputHalfWrapper, biasHalfWrapper);
	        outputHalfWrapper->copyToHost();
	        half * convh=(half*)outputHalfWrapper->getHostArray();
	        	for (int i =0;i<batchSize * dim.outputCubeSize;i++){
	        				output[i]=HalfToFloat(convh[i]);
	        	}
	        	outputWrapper->copyToDevice();
	    }


	StatefulTimer::timeCheck("Forward1::forward after call forward");


	//outputWrapper->copyToHost();

//    LOGI("////////////conv////////");
//
//    float * conv=(float*)outputWrapper->getHostArray();
//    int col=dim.outputSize;
//    float sum=0.0f;
//    for(int i =0;i<batchSize * dim.outputCubeSize; i++){
//        sum+=abs(temp[i]-conv[i]);
//    }
//    LOGI("diff %f",sum);

//    for(int i =0;i<10; i++){
//    	LOGI("temp[i]=%f conv[i]=%f ",temp[i],conv[i]);
//    }
    delete biasHalfWrapper;
    delete [] biasHalf;
	delete dataHalfWrapper;
	delete[] dataHalf;
	delete weightsHalfWrapper;
	delete[] weightsHalf;
	delete outputHalfWrapper;
	delete[] outputHalf;
}else{

	///////////////////////

	clock_t startTimer1, stopTimer1;
	startTimer1=clock();
    StatefulTimer::timeCheck("Forward1::forward START");
    kernel->in(batchSize);
    kernel->input(dataWrapper);
    kernel->input(weightsWrapper);
    kernel->output(outputWrapper);
LOGI("dim.outputCubeSize %d",dim.outputCubeSize);
    int globalSize = batchSize * dim.outputCubeSize;
    int workgroupsize = kernel->get_kernel_work_group_size();
    //int workgroupsize = std::min(globalSize, cl->getMaxWorkgroupSize());
    LOGI("workgroupsize %d, workgroupsize2 %d",workgroupsize, cl->getMaxWorkgroupSize());
    globalSize = (( globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;

    kernel->run_1d(globalSize, workgroupsize);
    cl->finish();
    StatefulTimer::timeCheck("Forward1::forward after call forward");

    if(dim.biased) {
        addBias->forward(
            batchSize, dim.numFilters, dim.outputSize,
            outputWrapper, biasWrapper);
    }
    StatefulTimer::timeCheck("Forward1::forward END");



stopTimer1 = clock();
double elapse = 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC;
LOGI("convolution took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

outputWrapper->copyToHost();

    	    float * conv=(float*)outputWrapper->getHostArray();
int col=dim.outputSize;

LOGI("2. dim.outputCubeSize %d",dim.outputCubeSize);
LOGI("2. dim.outputSizeSquared %d",dim.outputSizeSquared);

LOGI("col %d",dim.outputSize);
    	    LOGI("////////////conv////////");
    	    for (int i =0;i<10;i++){
    			string displayArraY="";
    			for (int j =0;j<10;j++){
    				displayArraY= displayArraY+ "-" + to_string_with_precision2(conv[i*col+j+1*dim.outputCubeSize]);
    			}
    			LOGI("%s",displayArraY.c_str());
    			displayArraY.clear();
    	    }
    	    LOGI("////////////conv////////");
//    	    float sum=0.0f;
//    	    for(int i =0;i<batchSize * dim.outputCubeSize; i++){
//    	    	//LOGI("temp[i]=%f conv[i]=%f ",temp[i],conv[i]);
//    	        sum+=abs(temp[i]-conv[i]);
//    	    }
//    	    LOGI("diff %f",sum);
}
//delete [] temp;
#endif
}

Forward1::Forward1(bool needToNormalize,int batchSize,EasyCL *cl, LayerDimensions dim)
        {

	this->cl=cl;
	this->dim=dim;

	normalization=needToNormalize;
	setup=false;
	#if TEST_FORWARD==1
    	addBias = new AddBias(cl);
    	LOGI("2)numFilters %d",dim.numFilters);

    std::string options = "";
    options += dim.buildOptionsString();

    // [[[cog
    // import stringify
    // stringify.write_kernel2("kernel", "cl/forward1.cl", "convolve_imagecubes_float2", 'options')
    // ]]]
    // generated using cog, from cl/forward1.cl:
    const char * kernelSource =
    	    "void kernel convolve_imagecubes_float2(\n"
    	    "    const int numExamples,\n"
    	    "      global const float *inputs, global const float *filters,\n"
    	    "    global float *output) {\n"
    	    "    int globalId = get_global_id(0);\n"
    	    "\n"
    	    "    int exampleId = (globalId / gOutputSizeSquared) / gNumFilters;\n"
    	    "    int filterId = (globalId / gOutputSizeSquared) % gNumFilters;\n"
    	    "\n"
    	    "    // intraimage coords\n"
    	    "    int outputRow = (globalId % gOutputSizeSquared) / gOutputSize;\n"
    	    "    int outputCol = (globalId % gOutputSizeSquared) % gOutputSize;\n"
    	    "\n"
    	    "\n"
    	    "    float sum = 0;\n"
    	    "    if (exampleId < numExamples) {\n"
    	    "        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {\n"
    	    "            global float const*inputPlane = inputs + exampleId * gNumInputPlanes * gInputSizeSquared + inputPlaneIdx * gInputSizeSquared;\n"
    	    "            global float const*filterPlane = filters + filterId * gNumInputPlanes * gFilterSizeSquared + inputPlaneIdx * gFilterSizeSquared;\n"
    	    "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    	    "                // trying to reduce register pressure...\n"
    	    "                #if gPadZeros == 1\n"
    	    "                    #define inputRowIdx (outputRow + u)\n"
    	    "                #else\n"
    	    "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    	    "                #endif\n"
    	    "                global float const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    	    "                global float const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    	    "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    	    "                #pragma unroll\n"
    	    "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    	    "                    #if gPadZeros == 1\n"
    	    "                        #define inputColIdx (outputCol + v)\n"
    	    "                    #else\n"
    	    "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    	    "                    #endif\n"
    	    "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    	    "                    if (process) {\n"
    	    "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    	    "                    }\n"
    	    "                }\n"
    	    "            }\n"
    	    "        }\n"
    		"		output[globalId] = sum;\n"
    	    "    }\n"
    	    "\n"
    	    "}\n"
    	    "\n"
    	    "";

    string operation="Forward1_"+std::to_string(dim.numFilters);
    kernel = cl->buildKernelFromString(operation, kernelSource, "convolve_imagecubes_float2", options, "../../cl/forward1.cl");


    const char * kernelSource2 =
    				"#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n\n"
    		        "void kernel convolve_imagecubes_half2(\n"
    		        "    const int numExamples,\n"
    		        "      global const half *inputs, global const half *filters,\n"
    		        "    global half *output) {\n"
    		        "    int globalId = get_global_id(0);\n"
    		        "\n"
    		        "    int outputImage2Id = globalId / gOutputSizeSquared;\n"
    		        "    int exampleId = outputImage2Id / gNumFilters;\n"
    		        "    int filterId = outputImage2Id % gNumFilters;\n"
    		        "\n"
    		        "    // intraimage coords\n"
    		        "    int localid = globalId % gOutputSizeSquared;\n"
    		        "    int outputRow = localid / gOutputSize;\n"
    		        "    int outputCol = localid % gOutputSize;\n"
    		        "\n"
    		        "    global half const*inputCube = inputs + exampleId * gNumInputPlanes * gInputSizeSquared;\n"
    		        "    global half const*filterCube = filters + filterId * gNumInputPlanes * gFilterSizeSquared;\n"
    		        "\n"
    		        "    half sum = 0;\n"
    		        "    if (exampleId < numExamples) {\n"
    		        "        for (int inputPlaneIdx = 0; inputPlaneIdx < gNumInputPlanes; inputPlaneIdx++) {\n"
    		        "            global half const*inputPlane = inputCube + inputPlaneIdx * gInputSizeSquared;\n"
    		        "            global half const*filterPlane = filterCube + inputPlaneIdx * gFilterSizeSquared;\n"
    		        "            for (int u = -gHalfFilterSize; u <= gHalfFilterSize - gEven; u++) {\n"
    		        "                // trying to reduce register pressure...\n"
    		        "                #if gPadZeros == 1\n"
    		        "                    #define inputRowIdx (outputRow + u)\n"
    		        "                #else\n"
    		        "                    #define inputRowIdx (outputRow + u + gHalfFilterSize)\n"
    		        "                #endif\n"
    		        "                global half const *inputRow = inputPlane + inputRowIdx * gInputSize;\n"
    		        "                global half const *filterRow = filterPlane + (u+gHalfFilterSize) * gFilterSize + gHalfFilterSize;\n"
    		        "                bool rowOk = inputRowIdx >= 0 && inputRowIdx < gInputSize;\n"
    		        "                #pragma unroll\n"
    		        "                for (int v = -gHalfFilterSize; v <= gHalfFilterSize - gEven; v++) {\n"
    		        "                    #if gPadZeros == 1\n"
    		        "                        #define inputColIdx (outputCol + v)\n"
    		        "                    #else\n"
    		        "                        #define inputColIdx (outputCol + v + gHalfFilterSize)\n"
    		        "                    #endif\n"
    		        "                    bool process = rowOk && inputColIdx >= 0 && inputColIdx < gInputSize;\n"
    		        "                    if (process) {\n"
    		        "                            sum += inputRow[inputColIdx] * filterRow[v];\n"
    		        "                    }\n"
    		        "                }\n"
    		        "            }\n"
    		        "        }\n"
    		        "    }\n"
    		        "\n"
    		        "    if (exampleId < numExamples) {\n"
    		        "        output[globalId] = sum;\n"
    		        "    }\n"
    		        "}\n"
    		        "\n"
    		        "";
        string operation2="ForwardHalf1_"+std::to_string(dim.numFilters);

        options=options+" -qcom-accelerate-16-bit -qcom-sched-rule=2";
        LOGI("option: %s ",options.c_str());
        kernelH = cl->buildKernelFromString(operation2, kernelSource2, "convolve_imagecubes_half2", options, "../../cl/forward1.cl");
#endif

//        string operation3="testForward1_"+std::to_string(dim.numFilters);
//    test = cl->buildKernelFromString(operation3, kernelSource, "convolve_imagecubes_float2", options, "../../cl/forward1.cl");

        buildKernelConvolve(batchSize);
}

void Forward1::buildKernelConvolve(int batchSize) {
	TemplatedKernel builder(cl);
	 //string identifier2="testForward1_"+std::to_string(dim.numFilters);

	string identifier2="testForward1_";
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
	if (not binariesManager->alreadyCompiledKernel("convolve_imagecubes_float2","",identifier2,filepath)){


	    setupBuilderConvolve(&builder, batchSize);
	}





        this->test = builder.buildKernel(
           		identifier2,
               "forward1.cl",
               getKernelTemplateConvolve(),
               "convolve_imagecubes_float2",
               false
        );
    }




void Forward1::setupBuilderConvolve(TemplatedKernel *builder,int batchSize) {

	string activationFunction("linear");
	string partialVectorizationType="default";
	string partialVectorizationLoad="default";
	string constantMemPartialVectorizationLoad="default";
	string initializationCondition="default";
	string internalLoopString1="default";
	string internalLoopString1norm="default";
	string internalLoopString2="default";
	string internalLoopStringNormalization="default";
	string internalLoopString1withPartialVectorization="default";
	bool fullvectorization=true;
	bool partialvectorization=true;
	bool ok1=true;
	int loop_count_partialVectorization=0;
	int remainerPartialVectorization=0;
	string initString="default";
	string dotString="default";
	string loop_string_partialVectorization="default";
	string extra_loop_string_partialVectorization="default";
	int vectorSize=0;
    string outputPoolingSelectorString="";
	string endPoolingString="}\n";
	string endPoolingString2="";
	string poolingSelectorString="";

	if (dim.useMaxPooling){
		setPoolingLayer(outputPoolingSelectorString,endPoolingString,endPoolingString2,poolingSelectorString,builder);
	}else{
		builder->set("gresult", "{{gActivationFunction}}");
		setActivationFunction(builder);
		setNonPoolingLayerVariable(builder,endPoolingString,endPoolingString2,fullvectorization);
	}


	testCondition(ok1);


	int countVectorizationPadding=dim.inputPlanes%4;
	if (countVectorizationPadding!=0)
		fullvectorization=false;


	if (partialvectorization){
		setAutoVectorization(vectorSize,remainerPartialVectorization,loop_count_partialVectorization,ok1, partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad,initializationCondition,builder,loop_string_partialVectorization, extra_loop_string_partialVectorization, initString, dotString);
	}

	setHintCompiler(batchSize,fullvectorization,partialvectorization,partialVectorizationType,builder);

//	if ((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0)))
//	if (normalization)
//		if (partialvectorization)
//			LOGI("internalLoopStringNormalization");
//		else
//			LOGI("internalLoopString1norm");
//	else
//		if(partialvectorization)
//			LOGI("internalLoopString1withPartialVectorization");
//		else
//			LOGI("internalLoopString1");
//	else
//		LOGI("internalLoopString2");

	setInternalLoop(ok1,loop_count_partialVectorization,internalLoopString1,internalLoopString1norm,internalLoopString2,internalLoopStringNormalization,internalLoopString1withPartialVectorization,initializationCondition,loop_string_partialVectorization,extra_loop_string_partialVectorization,partialVectorizationType,partialVectorizationLoad,constantMemPartialVectorizationLoad);

	writeKernelcode(builder, outputPoolingSelectorString,  poolingSelectorString,  partialvectorization,  normalization,  internalLoopStringNormalization,  internalLoopString1norm,  internalLoopString1withPartialVectorization,  internalLoopString1,  internalLoopString2, fullvectorization,  batchSize, ok1);


}

STATIC std::string Forward1::getKernelTemplateConvolve() {


	const char * kernelSource =
	    	    "void kernel {{gHintCompiler}}convolve_imagecubes_float2(\n"
	    	    "    const __global {{gVectorType}}* restrict inputs, __constant {{gVectorType}} *filters,\n"
	    	    "    global float *output {{gBias}} {{gNormalization}} {{gPoolingOutputSelector}}) {\n"
	    	    "    const int globalId = get_global_id(0);\n"
	    	    "\n"
	    	    "    int exampleId = ((globalId) {{DIVgOutputSizeSquared}}) / {{gNumFilters}};\n"
	    	    "    int filterId = ((globalId) {{DIVgOutputSizeSquared}}) % {{gNumFilters}};\n"
	    	    "\n"
				"    {{gVectorType}} sum = 0;\n"
				"    {{gMaxPooling}}"
	    	    "        {{gImageRowCoord}}"
	    	    "        {{gImageColCoord}}"
	    	    "        {{gBeginFirstLoop}}\n"
	    	    "            int inputPlaneIdx = exampleId * {{gNumInputPlanesTimeGInputSizeSquared}} {{gPlusInputPlaneIdxTimeGInputSizeSquared}};\n"
				"            int filterIdx=filterId * {{gNumInputPlanesTimeGFilterSizeSquared}} {{gPlusInputPlaneIdxTimeGFilterSizeSquared}};\n"
	    		"        {{gInternalLoop}}"
	    	    "      {{gEndFirstLoop}}\n"
				"\n"
				"    {{gMaxPoolingEnd}}\n"
	    		"    output[globalId] = {{gresult}};\n"
				"    {{outputPoolingSelector}}"
	    	    "\n"
	    	    "}\n"
	    	    "\n"
	    	    "";



    return kernelSource;
}




void Forward1::testCondition(bool &ok1){
	if(dim.outputSizeSquared==1){
			for (int u = -dim.halfFilterSize; u <= (dim.halfFilterSize-(dim.filterSize % 2 == 0 ? 1 : 0)); u++){
				int temp = u+dim.halfFilterSize;
				if ((temp < 0 ) || (temp >= dim.inputSize)){
					ok1=false;
					break;
				}
			}
	}else
	ok1=false;
}


void Forward1::setNonPoolingLayerVariable(TemplatedKernel *builder,string &endPoolingString,string &endPoolingString2,bool fullvectorization){

	builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
	builder->set("gSumAndBias", dim.biased? "{{gSum}}+bias[(globalId {{DIVgOutputSizeSquared}}) % {{gNumFilters}} ]":"{{gSum}}");
	builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((globalId % {{gOutputSizeSquared}}) % {{gOutputSize}})*"+to_string(dim.stride)+";\n":"int outputCol = (globalId % {{gOutputSizeSquared}}) % {{gOutputSize}};\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = ((globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}})*"+to_string(dim.stride)+";\n":"int outputRow = (globalId % {{gOutputSizeSquared}}) {{DIVgOutputSize}};\n":"");
	builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.outputSize):"");
	builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
	builder->set("gSum", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "dot( sum,(float4)(1.0f,1.0f,1.0f,1.0f))":"sum"));
	builder->set("gMaxPooling", "");
	builder->set("gMaxPoolingEnd", "");
}


void  Forward1::setActivationFunction(TemplatedKernel *builder){


		if (dim.activationLayer==1)
			builder->set("gActivationFunction", "{{gSumAndBias}}");
		if (dim.activationLayer==2)
			builder->set("gActivationFunction", "fmax ( {{gSumAndBias}} , 0 )");
		if (dim.activationLayer==3)
			builder->set("gActivationFunction", "tanh ( {{gSumAndBias}} )");
		if (dim.activationLayer==4)
			builder->set("gActivationFunction", "(1.7159f * tanh(0.66667f * {{gSumAndBias}}))");
		if (dim.activationLayer==5)
			builder->set("gActivationFunction", "(1.0f / (1 + exp(- ({{gSumAndBias}}))))");
		if (dim.activationLayer==6)
			builder->set("gActivationFunction", "fmin (fmax ( {{gSumAndBias}} , 0 ),(exp({{gSumAndBias}}) - 1))");

}


void Forward1::setPoolingLayer(    string &outputPoolingSelectorString,string &endPoolingString,string &endPoolingString2,string &poolingSelectorString, TemplatedKernel *builder){

	poolingSelectorString=", global int * selectorArray, global float *gradInput";

	if((dim.outputSize-dim.maxPool_spatialExtent)%dim.maxPool_strides==0){

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
								"      maxPool=fmax(maxPool,sum);\n"
								"      sum=0;\n";

		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string maxPoolingBegin="float maxPool=-999.99f;\n"
								"    int selectorID=100;\n"
						        "    #pragma unroll\n"
								"    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
								"      #pragma unroll\n"
								"      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";
		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}else{

		if (dim.biased){
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

		}else{
			endPoolingString+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";

			endPoolingString2+="sum={{gActivationFunction}};\n"
					"      gradInput[(filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + outputRow * "+to_string(dim.outputSize)+"+ outputCol)]=sum;\n"
					"      if ((p1<"+to_string(dim.maxPool_strides)+")&&(p2<"+to_string(dim.maxPool_strides)+")){\n"
					"      selectorID=select(selectorID,(p2 * "+to_string(dim.maxPool_strides)+" + p1),(isgreater(sum,maxPool) &&(outputRow<{{gOutputSize}})&&(outputCol<{{gOutputSize}})));\n"
					"      maxPool=fmax(maxPool,sum);\n"
					"      }\n"
					"      sum=0;\n";


		}
		builder->set("gEndFirstLoop",(dim.inputPlanes==1)? endPoolingString2:endPoolingString);
		setActivationFunction(builder);

		builder->set("gSumAndBias", dim.biased? "sum+bias[((filterId*"+to_string(dim.outputSizeSquared)+" +exampleId * "+to_string(dim.numFilters*dim.outputSizeSquared)+" + (outputRow) * "+to_string(dim.outputSize)+"+ (outputCol)) {{DIVgOutputSizeSquared2}}) % {{gNumFilters}} ]":"{{gSum}}");


		builder->set("gImageColCoord", (dim.outputSize!=1) ? (dim.stride!=1) ?"int outputCol = ((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride)+")*"+to_string(dim.maxPool_strides)+"+p1;\n":"int outputCol = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.maxPool_strides)+"+p1;\n":"");
		builder->set("gImageRowCoord", (dim.outputSizeSquared!=1) ? (dim.stride!=1) ? "int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+p2;\n":"int outputRow = (((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.maxPool_strides)+"+p2;\n":"");
		builder->set("DIVgOutputSize", (dim.outputSize!=1)? "/"+to_string(dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared", (dim.outputSizeSquared!=1)? "/"+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput):"");
		builder->set("DIVgOutputSizeSquared2", (dim.outputSizeSquared!=1)? "/"+to_string(dim.outputSizeSquared):"");
		builder->set("gSum", "sum");
		string extentString=to_string(dim.maxPool_spatialExtent);
		string extentPLUSRemainerString= to_string(dim.maxPool_spatialExtent+(dim.outputSize)%dim.maxPool_strides);
		string conditionString1="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) {{DIVgOutputSize}})*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString2="(((((globalId) % ("+to_string(dim.maxPool_sizeOutput*dim.maxPool_sizeOutput)+")) % ("+to_string(dim.maxPool_sizeOutput)+"))*"+to_string(dim.stride*dim.maxPool_strides)+"+"+to_string(dim.maxPool_strides)+")=="+to_string((dim.outputSize/dim.maxPool_strides)*dim.maxPool_strides)+")";
		string conditionString="("+conditionString1+"||"+conditionString2+")";

		string maxPoolingBegin="float maxPool=-999.99f;\n"
									   "    int selectorID=100;\n"
				                       "    #pragma unroll\n"
									   "    for(int p1=0;p1<"+to_string(dim.maxPool_spatialExtent)+";p1++){\n"
									   "      #pragma unroll\n"
									   "      for(int p2=0;p2<"+to_string(dim.maxPool_spatialExtent)+";p2++){\n";

		//note olivier: the next three commented line => special case if the maxpool size = odd nb and the remainer of the image size divided by the maxpoopl size is not equal to 0
		//note olivier: select dynamically the maxpooling size (for example 3 for all and 4 for the last one)
		// however it is really slow

//		string maxPoolingBegin="float maxPool=-999.99f;\n"
//							   "    int selectorID=100;\n"
//							   "    for(int p1=0;p1<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p1++){\n"
//							   "      for(int p2=0;p2<select("+extentString+","+extentPLUSRemainerString+","+conditionString+");p2++){\n";

		string maxPoolingEnd="  }\n}\n";
		builder->set("gMaxPooling", maxPoolingBegin);
		builder->set("gMaxPoolingEnd", maxPoolingEnd);
		builder->set("gresult", "maxPool");
		outputPoolingSelectorString="selectorArray[globalId]=selectorID;\n";
	}

}
void Forward1::setAutoVectorization(int &vectorSize,int &remainerPartialVectorization,int &loop_count_partialVectorization,bool ok1, string &partialVectorizationType,string& partialVectorizationLoad,string &constantMemPartialVectorizationLoad,string &initializationCondition,TemplatedKernel *builder,string &loop_string_partialVectorization, string &extra_loop_string_partialVectorization, string &initString, string &dotString){

		vector<string>indexOpencl;
		indexOpencl.push_back("0");
		indexOpencl.push_back("1");
		indexOpencl.push_back("2");
		indexOpencl.push_back("3");
		indexOpencl.push_back("4");
		indexOpencl.push_back("5");
		indexOpencl.push_back("6");
		indexOpencl.push_back("7");
		indexOpencl.push_back("8");
		indexOpencl.push_back("9");
		indexOpencl.push_back("a");
		indexOpencl.push_back("b");
		indexOpencl.push_back("c");
		indexOpencl.push_back("d");
		indexOpencl.push_back("e");
		indexOpencl.push_back("f");
		int size =dim.filterSize;
		int cpt=0;
		vectorSize=4;
		partialVectorizationType="float4";
		partialVectorizationLoad="(*((__global float4*)&";
		constantMemPartialVectorizationLoad="(*((__constant float4*)&";
		initString="(float4)(0.0f,0.0f,0.0f,0.0f)";
		dotString="(float4)(1.0f,1.0f,1.0f,1.0f)";

		loop_count_partialVectorization=floor((size)/4);
		remainerPartialVectorization=floor((size)%4);
		if ((not ok1)&&(loop_count_partialVectorization==1)){
			cpt=0;
			initializationCondition="";
			for(int i=-(dim.filterSize >> 1);i<(vectorSize-(dim.filterSize >> 1));i++){
				initializationCondition+="            conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"),0,1)));\n";
				cpt++;
			}
		}


if (loop_count_partialVectorization>=1){

		if ((not ok1)&&(loop_count_partialVectorization!=1)){
			initializationCondition="";

				loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
				loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(vectorSize);i++){
						loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}

				if (normalization)
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

				//loop_string_partialVectorization+= "               sum=dot( inputsV*filterV,"+dotString+");\n            }";
		}else
			if (ok1){

				if ((remainerPartialVectorization)<4){
					int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
					initializationCondition="";
					extra_loop_string_partialVectorization="";
					if (loop_count_partialVectorization>1){
						loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}; v < -{{gHalfFilterSize}}+"+to_string(loop_count_partialVectorization*vectorSize)+"; v+="+to_string(vectorSize)+"){\n";
						loop_string_partialVectorization+="              float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
														 "              float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

						loop_string_partialVectorization+= "              sum+=dot( (inputsV*filterV),"+dotString+");\n            }\n            ";
					}else{

						loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
						loop_string_partialVectorization+= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					}

					if (remainerPartialVectorization!=0){

						if ((ok1)&&((dim.filterSize >> 1)==1)){
							extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";

							extra_loop_string_partialVectorization+="             float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																 "             float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}else{
								extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
/////////////////////////
								if (loop_count_partialVectorization>1){
									extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}else{
									extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
																	 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
								}
/////////////////////////
								//extra_loop_string_partialVectorization+="             inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
								//									 "             filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";

								cpt=0;
								for(int i=0;i<(remainerPartialVectorization);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
										cpt++;
									}
								for(int i=remainerPartialVectorization;i<(vectorSize);i++){
										extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
										cpt++;
									}

								if (normalization)
									extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
								else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
							}
						}
				}else{

					initializationCondition="";
					extra_loop_string_partialVectorization="";
					loop_string_partialVectorization= "sum+=dot( (inputsV*filterV),"+dotString+");\n";

					if (remainerPartialVectorization!=0){
						extra_loop_string_partialVectorization=partialVectorizationType+" conditionVector;\n";
						extra_loop_string_partialVectorization+="          for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
							extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
															 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

							cpt=0;
							for(int i=0;i<(remainerPartialVectorization);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(1.0f);\n";
									cpt++;
								}
							for(int i=remainerPartialVectorization;i<(vectorSize);i++){
									extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
									cpt++;
								}

							if (normalization)
								extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
							else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

						}
				}


			}else{

				if (normalization){
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				}
				else{
					loop_string_partialVectorization= "            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx-{{gHalfFilterSize}})]));\n";
					loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";
				}
			}

		if ((not ok1)&&((remainerPartialVectorization)!=0)){


			if ((remainerPartialVectorization)<4){

				int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
				if (loop_count_partialVectorization>1){
					extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}else{
					extra_loop_string_partialVectorization="            inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
												 "            filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
				}

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

			}else{

				extra_loop_string_partialVectorization="for (int v = -{{gHalfFilterSize}}+"+to_string((int)((loop_count_partialVectorization)*vectorSize))+"; v < -{{gHalfFilterSize}}+"+to_string(((loop_count_partialVectorization))*vectorSize+remainerPartialVectorization)+"; v+="+to_string(vectorSize)+"){\n";
				extra_loop_string_partialVectorization+="            float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+v]));\n"
												 "            float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+v)]));\n";

				cpt=0;
				for(int i=0;i<(remainerPartialVectorization);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+v+1+"+to_string(i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i)+"+v),0,1)));\n";
						cpt++;
					}
				for(int i=remainerPartialVectorization;i<(vectorSize);i++){
						extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
						cpt++;
					}

				if (normalization)
					extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n            }";
				else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n            }";

			}
		}
	}else{

			initializationCondition="";//no need
			loop_string_partialVectorization="";//no need
			int v=-(dim.filterSize >> 1)+loop_count_partialVectorization*vectorSize;
			extra_loop_string_partialVectorization="           float4 inputsV = "+partialVectorizationLoad+" inputs[(inputPosition)+"+to_string(v)+"]));\n"
													 "           float4 filterV = "+constantMemPartialVectorizationLoad+" filters[(filterRowIdx+"+to_string(v)+")]));\n";
			cpt=0;
			for(int i=0;i<(remainerPartialVectorization);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx2}}+"+to_string(v+1+i)+",0,1)*clamp({{gInputSize}}-({{inputColIdx2}}+"+to_string(i+v)+"),0,1)));\n";
				cpt++;
			}
			for(int i=remainerPartialVectorization;i<(vectorSize);i++){
				extra_loop_string_partialVectorization+="               conditionVector.s"+indexOpencl.at(cpt)+"=(0.0f);\n";
				cpt++;
			}

			if (normalization)
				extra_loop_string_partialVectorization+= "sum+=dot( scale*(inputsV+translate)*filterV,conditionVector);\n";
			else extra_loop_string_partialVectorization+= "sum+=dot( inputsV*filterV,conditionVector);\n";

		}
	}


void Forward1::setHintCompiler(int batchSize,bool &fullvectorization,bool &partialvectorization,string &partialVectorizationType,TemplatedKernel *builder){
	int possibleGlobalSize = batchSize * dim.outputCubeSize;
	int possibleWorkgroupsize = std::min(possibleGlobalSize, cl->getMaxWorkgroupSize());

	string hintCompilerString="__attribute__((vec_type_hint(";
	if ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
		hintCompilerString+="float4";
	else{
		if ((not fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0)))
			hintCompilerString+="float";
		else
			if (partialvectorization)
				hintCompilerString+=partialVectorizationType;
			else hintCompilerString+="float";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}

void Forward1::setInternalLoop(bool ok1,int loop_count_partialVectorization,string &internalLoopString1,string& internalLoopString1norm,string &internalLoopString2,string &internalLoopStringNormalization,string &internalLoopString1withPartialVectorization,string initializationCondition,string loop_string_partialVectorization,string extra_loop_string_partialVectorization,string partialVectorizationType,string partialVectorizationLoad,string constantMemPartialVectorizationLoad){

	internalLoopString1="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += inputs[inputRow + {{inputColIdx}}] * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString1norm="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            int inputRow = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            #pragma unroll\n"
								"            for (int v = -{{gHalfFilterSize}}; v <= {{gHalfFilterSizeMinusGEven}}; v++) {\n"
								"               sum += scale*(inputs[inputRow + {{inputColIdx}}]+translate) * filters[filterRowIdx+v] {{gCondition}};\n"
								"            }\n"
								"        }\n";

	internalLoopString2="sum += inputs[inputPlaneIdx] * filters[filterIdx] {{gCondition}};\n";

	internalLoopStringNormalization="";
	if (not ok1)
		internalLoopStringNormalization+=partialVectorizationType+" conditionVector;\n";
	if (loop_count_partialVectorization!=1){

		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}else{
		internalLoopStringNormalization+="#pragma unroll\n"
									"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
									"            \n"+initializationCondition+"\n"
									"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
									"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
									"            "+loop_string_partialVectorization+""
									"            "+extra_loop_string_partialVectorization+""
									"        }\n";
	}
	internalLoopString1withPartialVectorization="";
	if (not ok1)
		internalLoopString1withPartialVectorization+=partialVectorizationType+" conditionVector;\n";
	if ((ok1)&&((dim.filterSize >> 1)==1)){
		internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}else{

	internalLoopString1withPartialVectorization+="#pragma unroll\n"
								"        for (int u = -{{gHalfFilterSize}}; u <= {{gHalfFilterSizeMinusGEven}}; u++) {\n"
								"            \n"+initializationCondition+"\n"
								"            int inputPosition = inputPlaneIdx + {{inputRowIdx}} * {{gInputSize}}+ {{inputColIdx2}};\n"
								"            int filterRowIdx=filterIdx+ (u+{{gHalfFilterSize}}) * {{gFilterSize}} + {{gHalfFilterSize}};\n"
								"            "+loop_string_partialVectorization+""
								"            "+extra_loop_string_partialVectorization+""
								"        }\n";
	}
}

void Forward1::writeKernelcode(TemplatedKernel *builder,string outputPoolingSelectorString, string poolingSelectorString, bool partialvectorization, bool normalization, string internalLoopStringNormalization, string internalLoopString1norm, string internalLoopString1withPartialVectorization, string internalLoopString1, string internalLoopString2,bool fullvectorization, int batchSize,bool ok1){

	builder->set("outputPoolingSelector", outputPoolingSelectorString);
	builder->set("gPoolingOutputSelector", poolingSelectorString);

	builder->set("gNormalization", normalization? "    ,\n float translate, float scale":"");
	builder->set("gVectorType",((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "float4":"float"));

	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gInternalLoop",((dim.outputSize!=1)||(((dim.filterSize >> 1)!=0)))? normalization? partialvectorization? internalLoopStringNormalization:internalLoopString1norm:partialvectorization? internalLoopString1withPartialVectorization:internalLoopString1:internalLoopString2);


	builder->set("gPlusInputPlaneIdxTimeGFilterSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gFilterSizeSquared}}");
	builder->set("gPlusInputPlaneIdxTimeGInputSizeSquared",(dim.inputPlanes==1)? "":"+ planeId * {{gInputSizeSquared}}");

	builder->set("gBeginFirstLoop",(dim.inputPlanes==1)? "":((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? "      #pragma unroll\n    for (int planeId = 0; planeId < "+to_string((dim.inputPlanes/4))+"; planeId++) {\n":"      #pragma unroll\n    for (int planeId = 0; planeId < {{gNumInputPlanes}}; planeId++) {\n"));
	builder->set("gCondition", ok1 ? "":(((dim.filterSize >> 1)!=0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2)) !=0))? "*((float)(clamp({{inputRowIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputRowIdx}},0,1)*clamp({{inputColIdx}}+1,0,1)*clamp({{gInputSize}}-{{inputColIdx}},0,1)))":"");
	builder->set("gHalfFilterSizeMinusGEven", (dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0));
	builder->set("gNumInputPlanesTimeGFilterSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? (dim.inputPlanes*dim.filterSizeSquared/4):dim.inputPlanes*dim.filterSizeSquared));
	builder->set("gNumInputPlanesTimeGInputSizeSquared", ((fullvectorization)&&(dim.outputSize==1)&&((dim.filterSize >> 1)==0)&&(((dim.filterSize >> 1)-(dim.filterSize % 2 == 0 ? 1 : 0) ==0))? ((dim.inputPlanes*dim.inputSizeSquared)/4):dim.inputPlanes*dim.inputSizeSquared));
	builder->set("gLimit", batchSize * dim.numFilters * dim.outputSize * dim.outputSize);

	builder->set("gBias", dim.biased? ", __constant  float * bias":" ");
	builder->set("gNumExamples", batchSize);
	builder->set("inputRowIdx", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputRowIdx2", (dim.outputSizeSquared!=1) ? dim.padZeros ? "(outputRow + u)" : "(outputRow + u + {{gHalfFilterSize}})": dim.padZeros ? "(u)" : "(u + {{gHalfFilterSize}})");
	builder->set("inputColIdx",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol + v)" : "(outputCol + v + {{gHalfFilterSize}})": dim.padZeros ? "(v)" : "(v + {{gHalfFilterSize}})");
	builder->set("inputColIdx2",  (dim.outputSize!=1) ? dim.padZeros ? "(outputCol)" : "(outputCol + {{gHalfFilterSize}})": dim.padZeros ? "" : "({{gHalfFilterSize}})");
    builder->set("gNumInputPlanes", dim.inputPlanes);
    builder->set("gInputPlanes", dim.inputPlanes);
    builder->set("gInputSize", dim.inputSize);
    builder->set("gInputSizeSquared", dim.inputSizeSquared);
    builder->set("gNumFilters", dim.numFilters);
    builder->set("gFilterSize", dim.filterSize);
    builder->set("gHalfFilterSize",  dim.filterSize >> 1);
    builder->set("gFilterSizeSquared", dim.filterSizeSquared);
    builder->set("gNumOutputPlanes", dim.numFilters);
    builder->set("gOutputPlanes", dim.numFilters);
	builder->set("gOutputSize", dim.outputSize);
    builder->set("gOutputSizeSquared", dim.outputSizeSquared);
    builder->set("gPadZeros", dim.padZeros ? 1 : 0);
    builder->set("gMargin", dim.padZeros ? dim.filterSize >> 1 : 0);
    builder->set("gEven", dim.filterSize % 2 == 0 ? 1 : 0);
    builder->set("gSkip", dim.skip);
}

