// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "BackpropWeightsNaive.h"
#include "../../EasyCL/util/StatefulTimer.h"
#include "../util/stringhelper.h"



using namespace std;

#undef STATIC
#define STATIC 

#undef VIRTUAL
#define VIRTUAL 

inline const char * const BoolToString(bool b)
{
  return b ? "true" : "false";
}
VIRTUAL BackpropWeightsNaive::~BackpropWeightsNaive() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: ~BackpropWeightsNaive");
#endif

#if TEST_KERNEL == 1
	if (dim.test)
		delete kernel;
#endif
    delete kernel2;
}
PUBLIC VIRTUAL void BackpropWeightsNaive::calcGradWeights(int batchSize, CLWrapper *gradOutputWrapper, CLWrapper *imagesWrapper, CLWrapper *gradWeightsWrapper, CLWrapper *gradBiasWrapper, CLWrapper *weightWrapper, CLWrapper *previousStepVectorWrapper, CLWrapper *biasWrapper, CLWrapper *previousStepBiasVectorWrapper){
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: calcGradWeights");
#endif

#if MEASURE_BACKWARD_PROP == 1
	clock_t startTimer1, stopTimer1;
	startTimer1=clock();

	float* temp= 0;
	float* temp2= 0;
	float *gradBias=0;
	float *grad=0;



	if (dim.test){

		StatefulTimer::instance()->timeCheck("BackpropWeightsNaive start");



		temp= new float[dim.filtersSize];
		temp2= new float[(dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes))];

		kernel
		   ->in(learningMultiplier)
		   ->in(batchSize)
		   ->in(gradOutputWrapper)
			->in(imagesWrapper)
		   ->inout(gradWeightsWrapper);
		if(dim.biased) {
			kernel->inout(gradBiasWrapper);
		}

		int globalSize0 = dim.filtersSize;

		int workgroupsize0 = kernel->get_kernel_work_group_size();//cl->getMaxWorkgroupSize();
		LOGI("------------------------globalSize %d",globalSize0);
		globalSize0 = ((globalSize0 + workgroupsize0 - 1) / workgroupsize0) * workgroupsize0;
		LOGI("------------------------globalSize %d workgroupsize %d",globalSize0,workgroupsize0);
		startTimer1=clock();
		kernel->run_1d(globalSize0, workgroupsize0);
		cl->finish();
		stopTimer1 = clock();
			LOGI("----------------------calculate weight took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;



		gradWeightsWrapper->copyToHost();
		gradBiasWrapper->copyToHost();
		gradBias=(float *)gradBiasWrapper->getHostArray();
		grad=(float *)gradWeightsWrapper->getHostArray();
		for (int i= 0; i < dim.filtersSize;i++)
			temp[i]=grad[i];
		for (int i= 0; i < (dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes));i++)
			temp2[i]=gradBias[i];
	}
#endif
if(method==1){
	if (not setup){

		if (dim.isConv){
			globalSize=batchSize*dim.filtersSize;
			workgroupsize = batchSize;
		}else{
			globalSize = dim.filtersSize;
			workgroupsize = kernel2->get_kernel_work_group_size();
		}
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSize2 %d",globalSize);
		#endif
		globalSize = ((globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSiz2e %d workgroupsize2 %d",globalSize,workgroupsize);
		#endif
		setup=true;
//		LOGI("wg size %d",workgroupsize);
//		LOGI("allowed %zu",kernel2->get_kernel_work_group_size());
		if (1){
			kernel2->input(dim.momentum);
			kernel2->input(dim.learning_rate);
			kernel2->inout(weightWrapper);
			kernel2->input(previousStepVectorWrapper);
			if (dim.biased){
				kernel2->input(biasWrapper);
				kernel2->input(previousStepBiasVectorWrapper);
			}
		}
		kernel2
       ->in(learningMultiplier)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper);
		#if MEASURE_BACKWARD_PROP==1
			kernel2->inout(gradWeightsWrapper);
			if(dim.biased) {
				kernel2->inout(gradBiasWrapper);
			}
		#endif

    if (dim.needToNormalize){
		kernel2->input(dim.translate);
		kernel2->input(dim.scale);
    }
	}
	//LOGI("------------------------globalSiz2e %d workgroupsize2 %d filter %d",globalSize,workgroupsize,dim.filtersSize);




	#if MEASURE_BACKWARD_PROP == 1
		stopTimer1 = clock();
		LOGI("----------------------extra 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
		LOGI("----------------------max size %d ",kernel2->get_kernel_work_group_size()) ;

		startTimer1=clock();
	#endif
    kernel2->run_1d(globalSize, workgroupsize);
}else{
	if (dim.isConv){//test
		//float* temp0 = new float [dim.filtersSize];
    	//CLWrapper * tempVarWrapper = (CLWrapper *)cl->wrap(dim.filtersSize, temp0);
    	//tempVarWrapper->createOnDevice();
		if (not setup){
			if (1){
				//kernelTest1->output(tempVarWrapper);
				kernelTest1->input(dim.momentum);
				kernelTest1->input(dim.learning_rate);
				kernelTest1->inout(weightWrapper);
				kernelTest1->input(previousStepVectorWrapper);
				if (dim.biased){
					kernelTest1->input(biasWrapper);
					kernelTest1->input(previousStepBiasVectorWrapper);
				}
			}
			kernelTest1
		   ->in(learningMultiplier)
		   ->in(gradOutputWrapper)
			->in(imagesWrapper);
			#if MEASURE_BACKWARD_PROP==1
				kernelTest1->inout(gradWeightsWrapper);
				if(dim.biased) {
					kernelTest1->inout(gradBiasWrapper);
				}
			#endif

			if (dim.needToNormalize){
				kernelTest1->input(dim.translate);
				kernelTest1->input(dim.scale);
			}


			if (dim.filtersSize<5000){
				workgroupsize=128;
				globalSize=((dim.filtersSize+127)/128)*128;
			}else{
				workgroupsize=1024;
				globalSize=((dim.filtersSize+1023)/1024)*1024;
			}
		}
		#if MEASURE_BACKWARD_PROP == 1
				struct timeval start, end;
				gettimeofday(&start, NULL);
				//LOGI("------------------------globalSiz3e %d workgroupsize3 %d kernel %d",globalSize2,1024,kernelTest1->get_kernel_work_group_size());
				//int globalSize2=((dim.filtersSize+1023)/1024)*1024;
				LOGI("------------------------globalSiz3e %d workgroupsize3 %d kernel %d",globalSize,workgroupsize,kernelTest1->get_kernel_work_group_size());
		#endif
		kernelTest1->run_1d(globalSize, workgroupsize);

		#if MEASURE_BACKWARD_PROP == 1
				cl->finish();

				gettimeofday(&end, NULL);
					/*Print the amount of time taken to execute*/
					LOGI("=====>>>>>%f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));
		#endif
			/*tempVarWrapper->copyToHost();
			float error=0.0f;
			for (int i= 0; i < 20;i++)
				LOGI("%f %f",temp[i],temp0[i]);
						for (int i= 0; i < dim.filtersSize;i++)
							error+=abs(temp[i]-temp0[i]);
						LOGI("---------New version calculate weight) error backprop_floats %f",error);
*/
			/*delete[] temp0;
			delete tempVarWrapper;*/
	}else{
		if (not setup){

		if (dim.isConv){
			globalSize=batchSize*dim.filtersSize;
			workgroupsize = batchSize;
		}else{
			globalSize = dim.filtersSize;
			workgroupsize = kernel2->get_kernel_work_group_size();
		}
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSize2 %d",globalSize);
		#endif
		globalSize = ((globalSize + workgroupsize - 1) / workgroupsize) * workgroupsize;
		#if MEASURE_BACKWARD_PROP == 1
			LOGI("------------------------globalSiz2e %d workgroupsize2 %d",globalSize,workgroupsize);
		#endif
		setup=true;
//		LOGI("wg size %d",workgroupsize);
//		LOGI("allowed %zu",kernel2->get_kernel_work_group_size());
		if (1){
			kernel2->input(dim.momentum);
			kernel2->input(dim.learning_rate);
			kernel2->inout(weightWrapper);
			kernel2->input(previousStepVectorWrapper);
			if (dim.biased){
				kernel2->input(biasWrapper);
				kernel2->input(previousStepBiasVectorWrapper);
			}
		}
		kernel2
       ->in(learningMultiplier)
       ->in(gradOutputWrapper)
        ->in(imagesWrapper);
		#if MEASURE_BACKWARD_PROP==1
			kernel2->inout(gradWeightsWrapper);
			if(dim.biased) {
				kernel2->inout(gradBiasWrapper);
			}
		#endif

		if (dim.needToNormalize){
			kernel2->input(dim.translate);
			kernel2->input(dim.scale);
		}
		}
		//LOGI("------------------------globalSiz2e %d workgroupsize2 %d filter %d",globalSize,workgroupsize,dim.filtersSize);




		#if MEASURE_BACKWARD_PROP == 1
			stopTimer1 = clock();
			LOGI("----------------------extra 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
			LOGI("----------------------max size %d ",kernel2->get_kernel_work_group_size()) ;

			startTimer1=clock();
		#endif
		kernel2->run_1d(globalSize, workgroupsize);
	}
}

	#if MEASURE_BACKWARD_PROP == 1
		stopTimer1 = clock();
		LOGI("----------------------calculate weight 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;

		startTimer1=clock();
		if (dim.test){
			gradWeightsWrapper->copyToHost();
			gradBiasWrapper->copyToHost();
			cl->finish();
			for (int i= 0; i < 20;i++)
				LOGI("%f %f",temp[i],grad[i]);
			float error=0.0f;
			for (int i= 0; i < dim.filtersSize;i++)
				error+=abs(temp[i]-grad[i]);
			LOGI("calculate weight) error backprop_floats %f",error);
			error=0.0f;
			for (int i= 0; i <(dim.filtersSize/(dim.filterSizeSquared*dim.inputPlanes));i++)
				error+=abs(temp2[i]-gradBias[i]);

			LOGI("error bias %f",error);
			StatefulTimer::instance()->timeCheck("BackpropWeightsNaive end");
			delete[] temp;
			delete[] temp2;
		}
		stopTimer1 = clock();
        LOGI("----------------------extra bis 2 took %g ms\n\n", 1000.0* (double)(stopTimer1 - startTimer1)/(double)CLOCKS_PER_SEC) ;
	#endif

}

float BackpropWeightsNaive::learningRateToMultiplier(int batchSize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/conv/BackpropWeightsNaive.cpp: learningRateToMultiplier");
#endif


    return 1.0f;
}

BackpropWeightsNaive::BackpropWeightsNaive(EasyCL *cl, LayerDimensions dim) :
        learningMultiplier(learningRateToMultiplier(dim.batchsize))
            {
	this->cl=cl;
	this->dim=dim;
    std::string options = dim.buildOptionsString();
    string imageV_with_possible_normalization="";
    string temp="";
    if (dim.needToNormalize){
    		 imageV_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 }else
    		 imageV_with_possible_normalization="upstreamResult";

#if TEST_KERNEL==1
    if (dim.test){
	   temp=
		"// Copyright Hugh Perkins 2014,2015 hughperkins at gmail\n"
		"//\n"
		"// This Source Code Form is subject to the terms of the Mozilla Public License,\n"
		"// v. 2.0. If a copy of the MPL was not distributed with this file, You can\n"
		"// obtain one at http://mozilla.org/MPL/2.0/.\n"
		"\n"
		"// expected defines:\n"
		"// BIASED (or not)\n"
		"\n"
		"// globalId: [outPlane][inputPlane][filterRow][filterCol]\n"
		"// per-thread iteration: [n][outputRow][outputCol]\n"
		"void kernel backprop_floats(const float learningRateMultiplier,\n"
		"        const int batchSize,\n"
		"         global const float *gradOutput, global const float *images,\n"
		"        global float *gradWeights\n"
		"        #ifdef BIASED\n"
		"            , global float *gradBiasWeights\n"
		"        #endif\n"
		" ) {\n"
		"    int globalId = get_global_id(0);\n"
		"    if (globalId >= gNumFilters * gInputPlanes * gFilterSize * gFilterSize) {\n"
		"        return;\n"
		"    }\n"
		"\n"
		"    int IntraFilterOffset = globalId % gFilterSizeSquared;\n"
		"    int filterRow = IntraFilterOffset / gFilterSize;\n"
		"    int filterCol = IntraFilterOffset % gFilterSize;\n"
		"\n"
		"    int filter2Id = globalId / gFilterSizeSquared;\n"
		"    int outPlane = filter2Id / gInputPlanes;\n"
		"    int upstreamPlane = filter2Id % gInputPlanes;\n"
		"\n"
		"    float thiswchange = 0;\n"
		"    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
		"    //       aggregate over:  [outRow][outCol][n]\n"
		"#ifdef BIASED\n"
		"    float thisbiaschange = 0;\n"
		"#endif\n"
		"    for (int n = 0; n < batchSize; n++) {\n"
		"        for (int outRow = 0; outRow < gOutputSize; outRow++) {\n"
		"            int upstreamRow = outRow - gMargin + filterRow;\n"
		"            for (int outCol = 0; outCol < gOutputSize; outCol++) {\n"
		"                int upstreamCol = outCol - gMargin + filterCol;\n"
		"                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < gInputSize\n"
		"                    && upstreamCol < gInputSize;\n"
		"                if (proceed) {\n"
		"                    int resultIndex = (( n * gNumFilters\n"
		"                              + outPlane) * gOutputSize\n"
		"                              + outRow) * gOutputSize\n"
		"                              + outCol;\n"
		"                    float error = gradOutput[resultIndex];\n"
		"                    int upstreamDataIndex = (( n * gInputPlanes\n"
		"                                     + upstreamPlane) * gInputSize\n"
		"                                     + upstreamRow) * gInputSize\n"
		"                                     + upstreamCol;\n"
		"                    float upstreamResult = images[upstreamDataIndex];\n"
		"                    float thisimagethiswchange = "+imageV_with_possible_normalization+" * error;\n"
		"                    thiswchange += thisimagethiswchange;\n"
		"    #ifdef BIASED\n"
		"                    thisbiaschange += error;\n"
		"    #endif\n"
		"                }\n"
		"            }\n"
		"        }\n"
		"    }\n"
		"    // gradWeights:     [outPlane][upstreamPlane][filterRow][filterCol]\n"
		"    //       aggregate over:  [outRow][outCol][n]\n"
		"    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
		"#ifdef BIASED\n"
		"    bool writeBias = upstreamPlane == 0 && filterRow == gMargin && filterCol == gMargin;\n"
		"    if (writeBias) {\n"
		"        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n"
		"    }\n"
		"#endif\n"
		"}\n"
		"\n"
		"\n"
		"\n"
		"";

    }
#endif

    string kernelSource2 =
        		            "void kernel backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
        		            "         global const float *gradOutput, global const float *images,\n"
        		            "        global float *gradWeights\n"
        		            "        {{gBiasDeclaration}}\n"
        		            " ) {\n"
        		            "    int globalId = get_global_id(0);\n"
        		            "\n"
        		            "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
        		            "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
        		            "\n"
        		            "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
        		            "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
        		            "\n"
        		            "    float thiswchange = 0;\n"
        		            "{{gBiasInit}}"
    						"    #pragma unroll\n"
        		            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
        		            "        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
        		            "            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
        		            "            for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
        		            "                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
        		            "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
        		            "                    && upstreamCol < {{gInputSize}};\n"
        		            "                if (proceed) {\n"
        		            "                    int resultIndex = (( n * {{gNumFilters}}\n"
        		            "                              + outPlane) * {{gOutputSize}}\n"
        		            "                              + outRow) * {{gOutputSize}}\n"
        		            "                              + outCol;\n"
        		            "                    float error = gradOutput[resultIndex];\n"
        		            "                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
        		            "                                     + upstreamPlane) * {{gInputSize}}\n"
        		            "                                     + upstreamRow) * {{gInputSize}}\n"
        		            "                                     + upstreamCol;\n"
        		            "                    float upstreamResult = images[upstreamDataIndex];\n"
        		            "                    float thisimagethiswchange = upstreamResult * error;\n"
        		            "                    thiswchange += thisimagethiswchange;\n"
        					"{{gBiasComputation}}"
        		            "                }\n"
        		            "            }\n"
        		            "        }\n"
        		            "    }\n"
        		            "    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
        		            "{{gBiasUpdate}}"
        		            "}\n"
        		            "";
#if TEST_KERNEL == 1
    if (dim.test){
    	const char * kernelSource =  temp.c_str();
		string operation="BackpropWeightsNaive"+std::to_string(dim.numFilters);
		kernel = cl->buildKernelFromString(operation, kernelSource, "backprop_floats", options, "cl/backpropweights.cl");
	}
#endif
    buildKernelBackward(kernelSource2);
}

void BackpropWeightsNaive::buildKernelBackward( string kernelSource) {

	string kernelSourceNew="";
	string kernelSourceNew2="";
	string gradComputeString2= "";
	setup=false;
    TemplatedKernel builder(cl);

       setupBuilderBackward(&builder);

        //string identifier2="BackpropWeightsNaive2"+std::to_string(dim.numFilters);

    	string identifier2="BackpropWeightsNaive2";
    		 identifier2=identifier2+"nbFilter=";
    		 identifier2=identifier2+std::to_string(dim.numFilters);
    		 identifier2=identifier2+"_InputSize="+std::to_string(dim.inputSize);
    		 identifier2=identifier2+"_batchsize="+std::to_string(dim.batchsize);
    		 identifier2=identifier2+"_OutputSize="+std::to_string(dim.outputSize);
    		 identifier2=identifier2+"_conv="+BoolToString(dim.isConv);
    		 identifier2=identifier2+"_normalize="+BoolToString(dim.needToNormalize);
    		 identifier2=identifier2+"_maxpool="+BoolToString(dim.useMaxPooling);

        if ((dim.filterSize==1)&&(dim.outputSize==1)&&(dim.padZeros == false)){
        	kernelSource=
        			            "void kernel {{gHintCompiler}} backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
        			            "         global const float *gradOutput, global const float *images\n"
        			            "{{gdeclareGradWeight}}"
        			            "        {{gBiasDeclaration}}\n"
        			            " ) {\n"
        			            "    int globalId = get_global_id(0);\n"
        			            "\n"
        			            "\n"
        			            "    int filter2Id = globalId;\n"
        			            "    int outPlane = filter2Id / {{gInputPlanes}};\n"
        			            "    int upstreamPlane = filter2Id % {{gInputPlanes}};\n"
        			            "\n"
        			            "    float thiswchange = 0;\n"
        			            "{{gBiasInit}}"
        						"    #pragma unroll\n"
        			            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
        			            "       float error = gradOutput[( n * {{gNumFilters}}+ outPlane)];\n"
        			            "       float upstreamResult = images[( n * {{gInputPlanes}}+ upstreamPlane)];\n"
        			            "       float thisimagethiswchange = upstreamResult * error;\n"
        			            "       thiswchange += thisimagethiswchange;\n"
        						"{{gBiasComputation}}"
        			            "    }\n"
        			            "    {{gradCompute}}"
								"    {{updateRule}}"
        			            "{{gBiasUpdate}}"
        			            "}\n"
        			            "\n"
        			            "\n"
        			            "\n"
        			            "";
         }else{
        	 if ((dim.outputSize==1)&&(dim.padZeros == false))
            	 kernelSource=
            			"void kernel {{gHintCompiler}} backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
     		            "         global const float *gradOutput, global const float *images\n"
     		            "{{gdeclareGradWeight}}"
     		            "        {{gBiasDeclaration}}\n"
     		            " ) {\n"
     		            "    int globalId = get_global_id(0);\n"
     		            "\n"
     		            "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
     		            "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
     		            "\n"
     		            "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
     		            "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
     		            "\n"
     		            "    float thiswchange = 0;\n"
     		            "{{gBiasInit}}"
            			"    #pragma unroll\n"
     		            "    for (int n = 0; n < {{gBatch}}; n++) {\n"
     		            "            int upstreamRow = filterRow;\n"
     		            "                int upstreamCol = filterCol;\n"
     		            "                    float error = gradOutput[( n * {{gNumFilters}}+ outPlane)];\n"
     		            "                    float upstreamResult = images[((( n * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol)];\n"
     		            "                    float thisimagethiswchange = upstreamResult * error;\n"
     		            "                    thiswchange += thisimagethiswchange;\n"
     					"{{gBiasComputation}}"
     		            "    }\n"
     		            "    {{gradCompute}}"
						"    {{updateRule}}"
     		            "{{gBiasUpdate}}"
     		            "}\n"
     		            "";

         }
        string declareGradWeightString = ",global float *gradWeights\n";
		 string gradComputeString = "gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n";

if (dim.isConv){
	gradComputeString="gradWeights[ globalId ] = learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
	gradComputeString2="gradWeights[ globalId ] = learningRateMultiplier * sumChange;\n";
	//LOGI("backward norm %d",dim.needToNormalize);

	 string imageV_with_possible_normalization= "";
	 if (dim.needToNormalize){
		 imageV_with_possible_normalization="(imageV+translate)*scale";//+"+to_string(dim.translate)+")*"+to_string(dim.scale);//
	 }else
		 imageV_with_possible_normalization="imageV";
	 string decaration_var_with_possible_normalization= "";
	 	 if (dim.needToNormalize)
	 		decaration_var_with_possible_normalization=" ,float translate, float scale";

	int remainer= dim.outputSize%4;
	int divider = dim.outputSize/4;
	string remainerString="";
	string remainerString2="";
	string remainerString3="";
	string addVariable="";
	if (remainer!=0){
			addVariable="    float4 thiswchangeV2= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"{{gBiasInit2}}";
			remainerString=
				"int outCol="+to_string(divider*4)+";\n"
				"       int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
				"       float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
				"       float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
				"       selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
				"       selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
				"       selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
				"       selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
				"       float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
				"       float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
				"       thiswchangeV2+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
				"{{gBiasComputationV2}}";

			switch(remainer) {
				case 1 : {
					remainerString2="+dot((float4)(1.0f,0.0f,0.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,0.0f,0.0f,0.0f),thisbiaschangeV2)";
					break;
				}
				case 2 : {
					remainerString2="+dot((float4)(1.0f,1.0f,0.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,1.0f,0.0f,0.0f),thisbiaschangeV2)";
					break;
				}
				case 3 : {
					remainerString2="+dot((float4)(1.0f,1.0f,1.0f,0.0f),thiswchangeV2)";
					remainerString3="+dot((float4)(1.0f,1.0f,1.0f,0.0f),thisbiaschangeV2)";
					break;
				}
			}
		}


	 kernelSource =
		"void kernel {{gHintCompiler}}backprop_floats({{updateVariable}} const float learningRateMultiplier,\n"
		"         global const float *gradOutput, global const float *images\n"
		"{{gdeclareGradWeight}}"
		"        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		" ) {\n"
		"    __local float shareArray[{{gBatch}}];\n"
		"    __local float shareArray2[{{gBatch}}];\n"
		"    int local_index = get_local_id(0);\n"
		"    int cpt=local_index%{{gBatch}};\n"
		"    int pos=local_index/{{gBatch}};\n"
		"    int globalId0 = get_global_id(0);\n"
		"    int globalId = get_global_id(0)/({{gBatch}});\n"
		"\n"
		"    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		"    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		"\n"
		"    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		"    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		"\n"
		"    float thiswchange = 0;\n"
		"    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"{{gBiasInit}}"
		"    "+addVariable+"\n"
		"    #pragma unroll\n"
		"    for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		"       int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		"       #pragma unroll\n"
		"       for (int outCol = 0; outCol < "+to_string(divider*4)+"; outCol=outCol+4) {\n"
		"           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
		"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
		"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
		"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
		"           selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
		"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
		"           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
		"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
		"{{gBiasComputationV}}"
		"       }\n"
		"       "+remainerString +"\n"
		"    }\n"
		"    shareArray[pos*{{gBatch}}+cpt]=dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV)"+remainerString2 +";\n"
		"    shareArray2[pos*{{gBatch}}+cpt]=dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +";\n"
		"    barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    for(unsigned int s = {{gHalfBatch}}; s > 0; s >>= 1){\n"
		"      if(cpt < s){\n"
		"        shareArray[pos*{{gBatch}}+cpt] += shareArray[pos*{{gBatch}}+cpt+ s];\n"
		"        shareArray2[pos*{{gBatch}}+cpt] += shareArray2[pos*{{gBatch}}+cpt + s];\n"
		"      }\n"
		"      barrier(CLK_LOCAL_MEM_FENCE);\n"
		"    }\n"
		"    if(cpt == 0){\n"
		"	   {{gradCompute}}"
		"      {{updateRule}}"
		"{{gBiasUpdate}}"
		"}\n"
		"}\n"
		"";


		// kernelSourceNew =
		// "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		// "         global const float *gradOutput, global const float *images\n"
		// "{{gdeclareGradWeight}}"
		// "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		// " ) {\n"
		// "    int local_index = get_local_id(0);\n"
		// "    int cpt=local_index%{{gBatch}};\n"
		// "    int pos=local_index/{{gBatch}};\n"
		// "    int globalId0 = get_global_id(0);\n"
		// "    int globalId = get_global_id(0)/({{gBatch}});\n"
		// "\n"
		// "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		// "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		// "\n"
		// "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		// "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		// "\n"
		// "    float thiswchange = 0;\n"
		// "    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
		// "{{gBiasInit}}"
		// "    "+addVariable+"\n"
		// "    int outRow = 0;\n"
		// "    //#pragma unroll\n"
		// "    //for (int outRow = 0; outRow < 8/*{{gOutputSize}}*/; outRow++) {\n"
		// "       int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		// "       //#pragma unroll\n"
		// "       for (int outCol = 0; outCol < "+to_string(divider*4)+"; outCol=outCol+4) {\n"
		// "           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		// "           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
		// "           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
		// "           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
		// "           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
		// "           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
		// "           selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
		// "           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
		// "           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
		// "           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
		// "{{gBiasComputationV}}"
		// "       }\n"
		// "       "+remainerString +"\n"
		// "    //}\n"
		// "    /*}*/\n"
        // "       tempVariables[pos]= dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV)"+remainerString2 +";\n"
		// "       tempVariables[pos]= dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +";\n"
		// "}\n"
		// "";

		// kernelSourceNew =
		 // "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 // "         global const float *gradOutput, global const float *images\n"
		 // "{{gdeclareGradWeight}}"
		 // "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 // " ) {\n"
		// "    __local float shareArray[{{gBatch}}];\n"
		// "    __local float shareArray2[{{gBatch}}];\n"
		// "    int local_index = get_local_id(0);\n"
		// "    int cpt=local_index%{{gBatch}};\n"
		// "    int pos=local_index/{{gBatch}};\n"
		// "    int globalId0 = get_global_id(0);\n"
		// "    int globalId = get_global_id(0)/({{gBatch}});\n"
		// "\n"
		// "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		// "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		// "\n"
		// "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		// "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		// "\n"
		// "    float thiswchange = 0;\n"
		// "    #pragma unroll\n"
		// "    for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		// "       int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		// "       #pragma unroll\n"
		// "       for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
		// "           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		// "           float error = gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]*select(0.0f,1.0f,(upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& upstreamCol < {{gInputSize}});\n"
		// "           float upstreamResult = images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol];\n"
		// "           thiswchange += upstreamResult * error;\n"
		// "       }\n"
		// "    }\n"
		// "    shareArray[pos*{{gBatch}}+cpt]=thiswchange;\n"
		// "    shareArray2[pos*{{gBatch}}+cpt]=0;\n"
		// "    barrier(CLK_LOCAL_MEM_FENCE);\n"
		// "    for(unsigned int s = {{gHalfBatch}}; s > 0; s >>= 1){\n"
		// "      if(cpt < s){\n"
		// "        shareArray[pos*{{gBatch}}+cpt] += shareArray[pos*{{gBatch}}+cpt+ s];\n"
		// "        shareArray2[pos*{{gBatch}}+cpt] += shareArray2[pos*{{gBatch}}+cpt + s];\n"
		// "      }\n"
		// "      barrier(CLK_LOCAL_MEM_FENCE);\n"
		// "    }\n"
		// "    if(cpt == 0){\n"
		// "	   gradWeights[ globalId ] = learningRateMultiplier * shareArray[pos*{{gBatch}}];\n"
		// "}\n"
		// "}\n"
		// "";

		 // string image_with_possible_normalization="";
         // if (dim.needToNormalize){
    		 // image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 // }else
    		 // image_with_possible_normalization="upstreamResult";


				// kernelSourceNew =
		 // "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 // "         global const float *gradOutput, global const float *images\n"
		 // "{{gdeclareGradWeight}}"
		 // "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 // " ) {\n"
		// "    int globalId = get_global_id(0)/({{gOutputSize}}*{{gOutputSize}});\n"
		// "    int outRow = (get_global_id(0)%({{gOutputSize}}*{{gOutputSize}})) / {{gOutputSize}};\n"
		// "    int outCol = (get_global_id(0)%({{gOutputSize}}*{{gOutputSize}})) % {{gOutputSize}};\n"
		// "\n"
		// "    int filterRow = (globalId % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		// "    int filterCol = (globalId % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		// "\n"
		// "    int outPlane = (globalId / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		// "    int upstreamPlane = (globalId / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		// "\n"
		// "    float thiswchange = 0;\n"
		// "{{gBiasInit0}}"
		// "    #pragma unroll\n"
		// "    for (int n = 0; n < {{gBatch}}/4; n++) {\n"
		// "            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		// "                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		// "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
		// "                    && upstreamCol < {{gInputSize}};\n"
		// "                if (proceed) {\n"
		// "                    int resultIndex = (( n * {{gNumFilters}}\n"
		// "                              + outPlane) * {{gOutputSize}}\n"
		// "                              + outRow) * {{gOutputSize}}\n"
		// "                              + outCol;\n"
		// "                    float error = gradOutput[resultIndex];\n"
		// "                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
		// "                                     + upstreamPlane) * {{gInputSize}}\n"
		// "                                     + upstreamRow) * {{gInputSize}}\n"
		// "                                     + upstreamCol;\n"
		// "                    float upstreamResult = images[upstreamDataIndex];\n"
		// "                    float thisimagethiswchange = "+image_with_possible_normalization+" * error;\n"
		// "                    thiswchange += thisimagethiswchange;\n"
		// "{{gBiasComputation}}"
		// "                }\n"
		// "    }\n"
		// "    gradWeights[ globalId ] = learningRateMultiplier * thiswchange;\n"
		// "{{gBiasUpdate0}}"
		// "}\n"
		// "";

		if (divider>0){
			kernelSourceNew =
			"void kernel {{gHintCompiler}}preCompute({{updateVariable}} const float learningRateMultiplier,\n"
			"         global const float *gradOutput, global const float *images\n"
			"{{gdeclareGradWeight}}"
			"        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
			" ) {\n"
			"    int globalId0 = get_global_id(0);\n"
			"\n"
			"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
			"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
			"\n"
			"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
			"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
			"\n"
			"    float thiswchange = 0;\n"
			"    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"{{gBiasInit}}"
			"    "+addVariable+"\n"
			"    //int outRow = 0;\n"
			"    #pragma unroll\n"
			"    for (int cpt = 0; cpt < {{gBatch}}; cpt++) {\n"
			"      #pragma unroll\n"
			"      for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
			"         int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
			"         #pragma unroll\n"
			"         for (int outCol = 0; outCol < "+to_string(divider*4)+"; outCol=outCol+4) {\n"
			"           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
			"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
			"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
			"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
			"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
			"           selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
			"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
			"           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
			"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
			"{{gBiasComputationV}}"
			"       }\n"
			"       "+remainerString +"\n"
			"    }\n"
			"    }\n"
			"{{gradCompute2}}"
			"{{updateRule2}}"
			"{{gBiasUpdate2}}"
			"}\n"
			"";

			 string gradComputeString2 = "       gradWeights[globalId0]= dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV)"+remainerString2 +";\n"
										 "{{gBiasUpdate1}}";
			 #if MEASURE_BACKWARD_PROP==1
					 (&builder)->set("gradCompute2", gradComputeString2);

			 #endif
			 #if MEASURE_BACKWARD_PROP==0
					 (&builder)->set("gradCompute2", "");
			 #endif

			string biasUpdateString1=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "        gradBiasWeights[outPlane] = learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +");\n"
						            "      }\n";


			(&builder)->set("gBiasUpdate1",biasUpdateString1);

		}else{

		 string image_with_possible_normalization="";
         if (dim.needToNormalize){
    		 image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 }else
    		 image_with_possible_normalization="upstreamResult";
					kernelSourceNew =
		 "void kernel {{gHintCompiler}}preCompute({{updateVariable}} const float learningRateMultiplier,\n"
		 "         global const float *gradOutput, global const float *images\n"
		 "{{gdeclareGradWeight}}"
		 "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 " ) {\n"
		"    int globalId0 = get_global_id(0);\n"
		"\n"
		"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		"\n"
		"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		"\n"
		"    float thiswchange = 0;\n"
		"{{gBiasInit0}}"
		"    #pragma unroll\n"
		"    for (int n = 0; n < {{gBatch}}; n++) {\n"
		"        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		"            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		"            for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
		"                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		"                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
		"                    && upstreamCol < {{gInputSize}};\n"
		"                if (proceed) {\n"
		"                    int resultIndex = (( n * {{gNumFilters}}\n"
		"                              + outPlane) * {{gOutputSize}}\n"
		"                              + outRow) * {{gOutputSize}}\n"
		"                              + outCol;\n"
		"                    float error = gradOutput[resultIndex];\n"
		"                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
		"                                     + upstreamPlane) * {{gInputSize}}\n"
		"                                     + upstreamRow) * {{gInputSize}}\n"
		"                                     + upstreamCol;\n"
		"                    float upstreamResult = images[upstreamDataIndex];\n"
		"                    float thisimagethiswchange = "+image_with_possible_normalization +" * error;\n"
		"                    thiswchange += thisimagethiswchange;\n"
		"{{gBiasComputation}}"
		"                }\n"
		"            }\n"
		"        }\n"
		"    }\n"
		"{{gradCompute2}} \n"
		"{{updateRule2}}"
		"{{gBiasUpdate2}}"
		"}\n"
		"";

		string gradComputeString2 = "    gradWeights[ globalId0 ] = learningRateMultiplier * thiswchange;\n"
									"{{gBiasUpdate0}}";
		#if MEASURE_BACKWARD_PROP==1
				(&builder)->set("gradCompute2", gradComputeString2);

		#endif
		#if MEASURE_BACKWARD_PROP==0
				(&builder)->set("gradCompute2", "");
		#endif

		/*kernelSourceNew =
			"void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
			"         global const float *gradOutput, global const float *images\n"
			"{{gdeclareGradWeight}}"
			"        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
			" ) {\n"
			"    int globalId0 = get_global_id(0);\n"
			"\n"
			"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
			"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
			"\n"
			"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
			"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
			"\n"
			"    float thiswchange = 0;\n"
			"    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"{{gBiasInit}}"
			"    "+addVariable+"\n"
			"    //int outRow = 0;\n"
			"    #pragma unroll\n"
			"    for (int cpt = 0; cpt < {{gBatch}}; cpt++) {\n"
			"      #pragma unroll\n"
			"      for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
			"         int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
			"         #pragma unroll\n"
			"         for (int outCol = 0; outCol < 1; outCol++) {\n"
			"           int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
			"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}+ outCol]));\n"
			"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
			"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
			"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
			"           selectV.s3=0.0f;//select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
			"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
			"           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
			"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
			"{{gBiasComputationV}}"
			"       }\n"
			"    }\n"
			"    }\n"
			"{{gradCompute2}}"
			"}\n"
			"";

			 string gradComputeString2 = "       gradWeights[globalId0]= dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV);\n"
										 "{{gBiasUpdate1}}";
			 #if MEASURE_BACKWARD_PROP==1
					 (&builder)->set("gradCompute2", gradComputeString2);

			 #endif
			 #if MEASURE_BACKWARD_PROP==0
					 (&builder)->set("gradCompute2", "");
			 #endif

			string biasUpdateString1=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "        gradBiasWeights[outPlane] = learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +");\n"
						            "      }\n";


			(&builder)->set("gBiasUpdate1",biasUpdateString1);*/

/* 			kernelSourceNew =
			"void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
			"         global const float *gradOutput, global const float *images\n"
			"{{gdeclareGradWeight}}"
			"        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
			" ) {\n"
			"    int globalId0 = get_global_id(0);\n"
			"\n"
			"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
			"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
			"\n"
			"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
			"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
			"\n"
			"    float thiswchange = 0;\n"
			"    float4 thiswchangeV= (float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"{{gBiasInit}}"
			"    "+addVariable+"\n"
			"    //int outRow = 0;\n"
			"    #pragma unroll\n"
			"    for (int cpt = 0; cpt < {{gBatch}}; cpt++) {\n"
			"      #pragma unroll\n"
			"      for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
			"         int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
			"           int upstreamCol = -{{gMargin}} + filterCol;\n"
			"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( cpt * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}]));\n"
			"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
			"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
			"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
			"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
			"           selectV.s3=0.0f;//select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
			"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
			"           float4 imageV= (*((__global float4*)&images[(( cpt * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
			"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
			"{{gBiasComputationV}}"
			"       }\n"
			"    }\n"
			"{{gradCompute2}}"
			"{{updateRule2}}"
			"{{gBiasUpdate2}}"
			"}\n"
			"";

			 string gradComputeString2 = "       gradWeights[globalId0]= dot((float4)(1.0f,1.0f,1.0f,0.0f),thiswchangeV);\n"
										 "{{gBiasUpdate1}}";
			 #if MEASURE_BACKWARD_PROP==1
					 (&builder)->set("gradCompute2", gradComputeString2);

			 #endif
			 #if MEASURE_BACKWARD_PROP==0
					 (&builder)->set("gradCompute2", "");
			 #endif

			string biasUpdateString1=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "        gradBiasWeights[outPlane] = learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV));\n"
						            "      }\n";


			(&builder)->set("gBiasUpdate1",biasUpdateString1); */




		// string image_with_possible_normalization="";
         // if (dim.needToNormalize){
    		 // image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 // }else
    		 // image_with_possible_normalization="upstreamResult";
					// kernelSourceNew =
		 // "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 // "         global const float *gradOutput, global const float *images\n"
		 // "{{gdeclareGradWeight}}"
		 // "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 // " ) {\n"
		// "    int globalId0 = get_global_id(0);\n"
		// "\n"
		// "    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		// "    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		// "\n"
		// "    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		// "    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		// "\n"
		// "    float thiswchange = 0;\n"
		// "{{gBiasInit0}}"
		// "    #pragma unroll\n"
		// "    for (int n = 0; n < {{gBatch}}; n++) {\n"
		// "        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		// "            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		// "              int upstreamCol = {{gMargin}} + filterCol;\n"
		// "              float2 gradOutputV = (*((__global float2*)&gradOutput[(( n * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}]));\n"
		// "              float2 selectV=(float2)(0.0f,0.0f);\n"
		// "              selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
		// "              selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
		// "              float2 errorV = (float2)(gradOutputV)*(float2)(selectV);\n"
		// "              float2 imageV= (*((__global float2*)&images[(( n * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
		// "              thiswchange+=dot((float2)(1.0f,1.0f),(float2)(imageV)*(float2)(errorV));\n"
		// "              upstreamCol = 2 - {{gMargin}} + filterCol;\n"
		// "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
		// "                    && upstreamCol < {{gInputSize}};\n"
		// "                if (proceed) {\n"
		// "                    int resultIndex = (( n * {{gNumFilters}}\n"
		// "                              + outPlane) * {{gOutputSize}}\n"
		// "                              + outRow) * {{gOutputSize}}\n"
		// "                              + 2;\n"
		// "                    float error = gradOutput[resultIndex];\n"
		// "                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
		// "                                     + upstreamPlane) * {{gInputSize}}\n"
		// "                                     + upstreamRow) * {{gInputSize}}\n"
		// "                                     + upstreamCol;\n"
		// "                    float upstreamResult = images[upstreamDataIndex];\n"
		// "                    float thisimagethiswchange = upstreamResult * error;\n"
		// "                    thiswchange += thisimagethiswchange;\n"
// //		"{{gBiasComputation}}"
	// //	"                }\n"
		// "            }\n"
		// "        }\n"
		// "    }\n"
		// "{{gradCompute2}} \n"
		// "{{updateRule2}}"
		// "{{gBiasUpdate2}}"
		// "}\n"
		// "";

//		string gradComputeString2 = "    gradWeights[ globalId0 ] = learningRateMultiplier * thiswchange;\n"
//									"{{gBiasUpdate0}}";
//		#if MEASURE_BACKWARD_PROP==1
//				(&builder)->set("gradCompute2", gradComputeString2);
//
//		#endif
//		#if MEASURE_BACKWARD_PROP==0
//				(&builder)->set("gradCompute2", "");
//		#endif

		/*string image_with_possible_normalization="";
         if (dim.needToNormalize){
    		 image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 }else
    		 image_with_possible_normalization="upstreamResult";
					kernelSourceNew =
		 "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 "         global const float *gradOutput, global const float *images\n"
		 "{{gdeclareGradWeight}}"
		 "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 " ) {\n"
		"    int globalId0 = get_global_id(0);\n"
		"\n"
		"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		"\n"
		"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		"\n"
		"    float thiswchange = 0;\n"
		"{{gBiasInit0}}"
		"    #pragma unroll\n"
		"    for (int n = 0; n < {{gBatch}}; n++) {\n"
		"        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		"            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		"            for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
		"                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		"                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
		"                    && upstreamCol < {{gInputSize}};\n"
		"                if (proceed) {\n"
		"                    int resultIndex = (( n * {{gNumFilters}}\n"
		"                              + outPlane) * {{gOutputSize}}\n"
		"                              + outRow) * {{gOutputSize}}\n"
		"                              + outCol;\n"
		"                    float error = gradOutput[resultIndex];\n"
		"                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
		"                                     + upstreamPlane) * {{gInputSize}}\n"
		"                                     + upstreamRow) * {{gInputSize}}\n"
		"                                     + upstreamCol;\n"
		"                    float upstreamResult = images[upstreamDataIndex];\n"
		"                    float thisimagethiswchange = "+image_with_possible_normalization +" * error;\n"
		"                    thiswchange += thisimagethiswchange;\n"
		"{{gBiasComputation}}"
		"                }\n"
		"            }\n"
		"        }\n"
		"    }\n"
		"{{gradCompute2}} \n"
		"{{updateRule2}}"
		"{{gBiasUpdate2}}"
		"}\n"
		"";

		string gradComputeString2 = "    gradWeights[ globalId0 ] = learningRateMultiplier * thiswchange;\n"
									"{{gBiasUpdate0}}";
		#if MEASURE_BACKWARD_PROP==1
				(&builder)->set("gradCompute2", gradComputeString2);

		#endif
		#if MEASURE_BACKWARD_PROP==0
				(&builder)->set("gradCompute2", "");
		#endif*/

		/*string image_with_possible_normalization="";
         if (dim.needToNormalize){
    		 image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 }else
    		 image_with_possible_normalization="upstreamResult";
					kernelSourceNew =
		 "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 "         global const float *gradOutput, global const float *images\n"
		 "{{gdeclareGradWeight}}"
		 "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 " ) {\n"
		"    int globalId0 = get_global_id(0);\n"
		"\n"
		"    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		"    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		"\n"
		"    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		"    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		"\n"
		"    float4 thiswchangeV = (float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"    float4 thisbiaschangeV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"    #pragma unroll\n"
		"    for (int n = 0; n < {{gBatch}}; n++) {\n"
		"        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		"            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		"           int upstreamCol = {{gMargin}} + filterCol;\n"
		"           float4 gradOutputV = (*((__global float4*)&gradOutput[(( n * {{gNumFilters}}+ outPlane) * {{gOutputSize}}+ outRow) * {{gOutputSize}}]));\n"
		"           float4 selectV=(float4)(0.0f,0.0f,0.0f,0.0f);\n"
		"           selectV.s0=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol < {{gInputSize}})));\n"
		"           selectV.s1=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+1 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+1 < {{gInputSize}})));\n"
		"           selectV.s2=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+2 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+2 < {{gInputSize}})));\n"
		"           selectV.s3=select(0.0f,1.0f,((upstreamRow >= 0) && (upstreamCol+3 >= 0) && (upstreamRow < {{gInputSize}})&& (upstreamCol+3 < {{gInputSize}})));\n"
		"           float4 errorV = (float4)(gradOutputV)*(float4)(selectV);\n"
		"           float4 imageV= (*((__global float4*)&images[(( n * {{gInputPlanes}}+ upstreamPlane) * {{gInputSize}}+ upstreamRow) * {{gInputSize}}+ upstreamCol]));\n"
		"           thiswchangeV+=(float4)("+imageV_with_possible_normalization +")*(float4)(errorV);\n"
		"           thisbiaschangeV += errorV;"
		"        }\n"
		"    }\n"
		"{{gradComputeTest}} \n"
		"}\n"
		"";

		string gradComputeString2 = "    gradWeights[ globalId0 ] = learningRateMultiplier * dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV);\n"
									"	 bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "    if (writeBias) {\n"
						            "      gradBiasWeights[outPlane] = learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV));\n"
						            "    }\n";
		#if MEASURE_BACKWARD_PROP==1
				(&builder)->set("gradComputeTest", gradComputeString2);

		#endif
		#if MEASURE_BACKWARD_PROP==0
				(&builder)->set("gradComputeTest", "");
		#endif*/

		// string image_with_possible_normalization="";
         // if (dim.needToNormalize){
    		 // image_with_possible_normalization="(upstreamResult+"+to_string(dim.translate)+")*"+to_string(dim.scale);
    	 // }else
    		 // image_with_possible_normalization="upstreamResult";
					// kernelSourceNew =
		 // "void kernel {{gHintCompiler}}preCompute(global float *tempVariables,{{updateVariable}} const float learningRateMultiplier,\n"
		 // "         global const float *gradOutput, global const float *images\n"
		 // "{{gdeclareGradWeight}}"
		 // "        {{gBiasDeclaration}}"+decaration_var_with_possible_normalization+"\n"
		 // " ) {\n"
		// "    int globalId0 = get_global_id(0);\n"
		// "\n"
		// "    int filterRow = (globalId0 % {{gFilterSizeSquared}}) / {{gFilterSize}};\n"
		// "    int filterCol = (globalId0 % {{gFilterSizeSquared}}) % {{gFilterSize}};\n"
		// "\n"
		// "    int outPlane = (globalId0 / {{gFilterSizeSquared}}) / {{gInputPlanes}};\n"
		// "    int upstreamPlane = (globalId0 / {{gFilterSizeSquared}}) % {{gInputPlanes}};\n"
		// "\n"
		// "    float thiswchange = 0;\n"
		// "{{gBiasInit0}}"
		// "    #pragma unroll\n"
		// "    for (int n = 0; n < {{gBatch}}; n++) {\n"
		// "        for (int outRow = 0; outRow < {{gOutputSize}}; outRow++) {\n"
		// "            int upstreamRow = outRow - {{gMargin}} + filterRow;\n"
		// "            for (int outCol = 0; outCol < {{gOutputSize}}; outCol++) {\n"
		// "                int upstreamCol = outCol - {{gMargin}} + filterCol;\n"
		// "                bool proceed = upstreamRow >= 0 && upstreamCol >= 0 && upstreamRow < {{gInputSize}}\n"
		// "                    && upstreamCol < {{gInputSize}};\n"
		// "                if (proceed) {\n"
		// "                    int resultIndex = (( n * {{gNumFilters}}\n"
		// "                              + outPlane) * {{gOutputSize}}\n"
		// "                              + outRow) * {{gOutputSize}}\n"
		// "                              + outCol;\n"
		// "                    float error = gradOutput[resultIndex];\n"
		// "                    int upstreamDataIndex = (( n * {{gInputPlanes}}\n"
		// "                                     + upstreamPlane) * {{gInputSize}}\n"
		// "                                     + upstreamRow) * {{gInputSize}}\n"
		// "                                     + upstreamCol;\n"
		// "                    float upstreamResult = images[upstreamDataIndex];\n"
		// "                    float thisimagethiswchange = "+image_with_possible_normalization +" * error;\n"
		// "                    thiswchange += thisimagethiswchange;\n"
		// "{{gBiasComputation}}"
		// "                }\n"
		// "            }\n"
		// "        }\n"
		// "    }\n"
		// "{{gradCompute2}} \n"
		// "{{updateRule2}}"
		// "{{gBiasUpdate2}}"
		// "}\n"
		// "";

		// string gradComputeString2 = "    gradWeights[ globalId0 ] = learningRateMultiplier * thiswchange;\n"
									// "{{gBiasUpdate0}}";
		// #if MEASURE_BACKWARD_PROP==1
				// (&builder)->set("gradCompute2", gradComputeString2);

		// #endif
		// #if MEASURE_BACKWARD_PROP==0
				// (&builder)->set("gradCompute2", "");
		// #endif
	}

	string updateWeight2=" ";
	string updateBiasWeight2="";
	string biasUpdateString2="    if (upstreamPlane == 0) {\n"
								"        {{updateBiasWeights2}}"
								"    }\n";;
	setupUpdateWeightVectorized(updateWeight2, updateBiasWeight2,remainerString2,remainerString3,(divider>0));
	(&builder)->set("updateRule2",updateWeight2);
	(&builder)->set("gBiasUpdate2",biasUpdateString2);
	(&builder)->set("updateBiasWeights2",updateBiasWeight2);

}

#if MEASURE_BACKWARD_PROP==1
		(&builder)->set("gdeclareGradWeight", declareGradWeightString);
#endif
#if MEASURE_BACKWARD_PROP==0
		(&builder)->set("gdeclareGradWeight", "");
#endif


#if MEASURE_BACKWARD_PROP==1
		(&builder)->set("gradCompute", gradComputeString);
		(&builder)->set("gdeclareGradWeight2", declareGradWeightString);
#endif
#if MEASURE_BACKWARD_PROP==0
		(&builder)->set("gradCompute", "");
		(&builder)->set("gdeclareGradWeight2", "");
#endif

        this->kernel2 = builder.buildKernel(
           		identifier2,
               "backprop_floats",
               kernelSource.c_str(),
               "backprop_floats",
               false
        );
        if (dim.isConv){
			this->kernelTest1 = builder.buildKernel(
					identifier2,
				   "preCompute",
				   kernelSourceNew.c_str(),
				   "preCompute",
				   false
			);
			// this->kernelTest2 = builder.buildKernel(
					// identifier2,
				   // "mergeSum",
				   // kernelSourceNew2.c_str(),
				   // "mergeSum",
				   // false
			// );
        }
    }

void BackpropWeightsNaive::setHintCompiler(int batchSize, TemplatedKernel *builder){
	int possibleGlobalSize = batchSize*dim.filtersSize;
	int possibleWorkgroupsize =  batchSize;//kernel2->get_kernel_work_group_size();//cl->getMaxWorkgroupSize();

	possibleGlobalSize = ((possibleGlobalSize + possibleWorkgroupsize - 1) / possibleWorkgroupsize) * possibleWorkgroupsize;

	string hintCompilerString="__attribute__((vec_type_hint(";
	if (dim.isConv)
		hintCompilerString+="float4";
	else{
		hintCompilerString+="float";
	}

	hintCompilerString+="))) __attribute__((work_group_size_hint("+to_string(possibleWorkgroupsize)+", 1, 1))) ";

	builder->set("gHintCompiler", hintCompilerString);
}
void BackpropWeightsNaive::setupBuilderBackward(TemplatedKernel *builder) {

	string updateWeight=" ";
	string updateWeight2=" ";
	string updateBiasWeights=" ";
	string defineVariableUpdates=" ";
	string updateBiasWeight=" ";
	string updateBiasWeight2=" ";
	string gradBiasComputeString0="";
	string biasUpdateString0="";
	string biasUpdateString1="";
	string gradBiasComputeString1="";


	//setupUpdateWeightVectorized(updateWeight2, updateBiasWeight2);
	setupUpdateWeight(updateWeight,updateBiasWeights, defineVariableUpdates,updateBiasWeight);

	builder->set("updateVariable",defineVariableUpdates);
	if (dim.isConv){
		updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
	}else
		updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thiswchange;\n";

	builder->set("updateRule",updateWeight);


	setHintCompiler(dim.batchsize, builder);
	builder->set("gHalfBatch",dim.batchsize>>1);
	builder->set("gBatch",dim.batchsize);
	builder->set("gInputSizeSquared",square(dim.inputSize));
	builder->set("gInputSize",dim.inputSize);
	builder->set("gInputPlanes",dim.inputPlanes);
	builder->set("gMargin",dim.padZeros ? dim.filterSize >> 1 : 0);
	builder->set("gOutputSize",dim.outputSize);
	builder->set("gFilterSize",dim.filterSize);
	builder->set("gFilterSizeSquared",dim.filterSize*dim.filterSize);
	builder->set("gNumFilters",dim.numFilters);

	string gradBiasComputeString="";

	if (dim.biased){
		#if MEASURE_BACKWARD_PROP==1
			builder->set("gBiasDeclaration",", global float *gradBiasWeights");
		#endif
		#if MEASURE_BACKWARD_PROP==0
			builder->set("gBiasDeclaration","");
		#endif

		//builder->set("gBiasDeclaration",", global float *gradBiasWeights");
		builder->set("gBiasInit","    float thisbiaschange = 0;\n");
		string biasUpdateString=
							"    bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
							"    if (writeBias) {\n"
							"{{gradBiasCompute}}"
							"        {{updateBiasWeights}}"
							"    }\n";
		if (dim.isConv){

			builder->set("gBiasInit","    float4 thisbiaschangeV = (float4)(0.0f,0.0f,0.0f,0.0f);\n");
			builder->set("gBiasComputationV","           thisbiaschangeV += errorV;\n");
			builder->set("gBiasInit2","    float4 thisbiaschangeV2 = (float4)(0.0f,0.0f,0.0f,0.0f);\n");
			builder->set("gBiasComputationV2","       thisbiaschangeV2 += errorV;\n");
			builder->set("gBiasComputation","           thisbiaschange += error;\n");
			builder->set("gBiasInit0","    float thisbiaschange = 0;\n");
			biasUpdateString=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "{{gradBiasCompute}}"
									"          {{updateBiasWeights}}"
						            "      }\n";

			biasUpdateString0=
									"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
						            "      if (writeBias) {\n"
						            "{{gradBiasCompute0}}"
						            "      }\n";
			//biasUpdateString1=
			//						"	   bool writeBias = upstreamPlane == 0 && filterRow == {{gMargin}} && filterCol == {{gMargin}};\n"
			//			            "      if (writeBias) {\n"
			//			            "{{gradBiasCompute1}}"
			//			            "      }\n";


			gradBiasComputeString="        gradBiasWeights[outPlane] = learningRateMultiplier * shareArray2[pos*{{gBatch}}];\n";
			gradBiasComputeString0="        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n";
			//gradBiasComputeString1="        gradBiasWeights[outPlane] = learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +");\n";
		}else{
			builder->set("gBiasComputation","                    thisbiaschange += error;\n");
			if ((dim.filterSize==1)&&(dim.outputSize==1)&&(dim.padZeros == false))
				biasUpdateString=
								"    if (upstreamPlane == 0) {\n"
								"{{gradBiasCompute}}"
								"        {{updateBiasWeights}}"
								"    }\n";
				gradBiasComputeString="        gradBiasWeights[outPlane] = learningRateMultiplier * thisbiaschange;\n";
		}
		builder->set("gBiasUpdate",biasUpdateString);
		builder->set("updateBiasWeights",updateBiasWeight);
		//builder->set("updateBiasWeights2",updateBiasWeight2);
		builder->set("gradBiasCompute0", gradBiasComputeString0);
		//builder->set("gradBiasCompute1", gradBiasComputeString1);
		builder->set("gBiasUpdate0",biasUpdateString0);
	}

	#if MEASURE_BACKWARD_PROP==1
		builder->set("gradBiasCompute", gradBiasComputeString);

	#endif
	#if MEASURE_BACKWARD_PROP==0
		builder->set("gradBiasCompute", "");
	#endif

}

void BackpropWeightsNaive::setupUpdateWeight(string &updateWeight,string &updateBiasWeights, string &defineVariableUpdates, string & updateBiasWeight){

	if (1){//only  SGD

		if (dim.isConv){
			updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray[pos*{{gBatch}}];\n";
		}else
			updateWeight="weight[ globalId ]=weight[ globalId ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thiswchange;\n";

		if (dim.biased){
			defineVariableUpdates="const float momentum, const float learning_rate, global float* weight, global float * pastTimeStepVector, global float* bias, global float * pastTimeStepBiasVector,";
			if (dim.isConv){
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * shareArray2[pos*{{gBatch}}];\n";
			}else
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepVector[globalId]-learning_rate*learningRateMultiplier * thisbiaschange;\n";

		}else
			defineVariableUpdates="const float momentum, const float learning_rate, global float* weight, global float * pastTimeStepVector,";
	}
}

void BackpropWeightsNaive::setupUpdateWeightVectorized(string &updateWeight, string & updateBiasWeight,string remainerString2,string remainerString3,bool selector){
//only for conv

	if (1){//only  SGD
		if (selector){
			updateWeight="weight[ globalId0 ]=weight[ globalId0 ]-momentum* pastTimeStepVector[globalId0]-learning_rate*learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thiswchangeV)"+remainerString2+");\n";

			if (dim.biased){
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepBiasVector[outPlane]-learning_rate*learningRateMultiplier * (dot((float4)(1.0f,1.0f,1.0f,1.0f),thisbiaschangeV)"+remainerString3 +");\n";
			}
		}else{
			updateWeight="weight[ globalId0 ]=weight[ globalId0 ]-momentum* pastTimeStepVector[globalId0]-learning_rate*learningRateMultiplier * thiswchange;\n";

			if (dim.biased){
				updateBiasWeight="bias[ outPlane ]=bias[ outPlane ]-momentum* pastTimeStepBiasVector[outPlane]-learning_rate*learningRateMultiplier * thisbiaschange;\n";
			}
		}
	}
}
