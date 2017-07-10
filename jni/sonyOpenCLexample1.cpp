//
//  sonyOpenCLexample1.cpp
//  OpenCL Example1
//
//  Created by Rasmusson, Jim on 18/03/13.
//
//  Copyright (c) 2013, Sony Mobile Communications AB
//  All rights reserved.
//
//  Redistribution and use in source and binary forms, with or without
//  modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of Sony Mobile Communications AB nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
//  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <stdio.h>
#include "sonyOpenCLexample1.h"

#include <android/bitmap.h>
#include <jni.h>


#include "trainEngine/train.h"
#include "predictEngine/predict.h"



#include "kernelManager/kernelManager.h"

#include <sys/stat.h>
#include <dirent.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

//#include "train.h"
#define CLMATH_VERBOSE 1
#define DEEPCL_VERBOSE 1


char* ReadFile(char *filename)
{
   char *buffer = NULL;
   int string_size, read_size;
   FILE *handler = fopen(filename, "r");
   LOGI( "fopen");

   if (handler)
   {
	   LOGI( "handler");
       // Seek the last byte of the file
       fseek(handler, 0, SEEK_END);
       LOGI( "Seek");
       // Offset from the first to the last byte, or in other words, filesize
       string_size = ftell(handler);
       // go back to the start of the file
       rewind(handler);

       // Allocate a string that can hold it all
       buffer = (char*) malloc(sizeof(char) * (string_size + 1) );

       // Read it all in one operation
       read_size = fread(buffer, sizeof(char), string_size, handler);

       // fread doesn't set it so put a \0 in the last position
       // and buffer is now officially a string
       buffer[string_size] = '\0';

       if (string_size != read_size)
       {
           // Something went wrong, throw away the memory and set
           // the buffer to NULL
           free(buffer);
           buffer = NULL;
       }

       // Always remember to close the file.
       fclose(handler);
    }

    return buffer;
}

int
mkpath(std::string s,mode_t mode)
{
    size_t pre=0,pos;
    std::string dir;
    int mdret;

    if(s[s.size()-1]!='/'){
        // force trailing / so we can handle everything in loop
        s+='/';
    }

    while((pos=s.find_first_of('/',pre))!=std::string::npos){
        dir=s.substr(0,pos++);
        pre=pos;
        if(dir.size()==0) continue; // if leading / first time is 0 length
        if((mdret=mkdir(dir.c_str(),mode)) && errno!=EEXIST){
            return mdret;
        }
    }
    return mdret;
}

int64_t getTimeNsec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000000000LL + now.tv_nsec;
}
static double TimeSpecToSeconds(struct timespec* ts)
{
    return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_runOpenCL(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{

//	 boost::iostreams::mapped_file_sink file_sink;
//	 boost::iostreams::mapped_file_params param_sink;
//	 param_sink.path = "/data/data/com.sony.openclexample1/preloadingData/result.bin";
//	 param_sink.offset = 0;
//	 param_sink.new_file_size = 4*10;
//	 file_sink.open(param_sink);
//	 char * s=(char *)file_sink.data();
//	 s[0]='j';
//	 s[1]='e';
//	 file_sink.close();
//    boost::iostreams::mapped_file_params params_;
//        boost::iostreams::mapped_file_sink sink_;
//        params_.length = 1;
//        params_.new_file_size = 1024;
//        params_.path = "/data/data/com.sony.openclexample1/preloadingData/weightstjefaisuntest.dat";
//        sink_.open(params_);
//        sink_.close();



	 struct timespec start1;
		struct timespec end1;
		double elapsedSeconds1;
		clock_gettime(CLOCK_MONOTONIC, &start1);


		struct timeval start, end;
		/*get the start time*/
		gettimeofday(&start, NULL);
////	int mkdirretval;
////	    mkdirretval=mkpath("/data/data/com.sony.openclexample1/app_execdir",0755);
////	    if (-1 == mkdirretval)
////	    {
////	        LOGI("Error creating directory!n");
////	        exit(1);
////	    }
//
////	 DIR *theFolder = opendir("/data/data/com.sony.openclexample1/app_execdir/binariesKernel/");
////	    struct dirent *next_file;
////	    char filepath[256];
////
////	    while ( (next_file = readdir(theFolder)) != NULL )
////	    {
////	        // build the path for each file in the folder
////	        sprintf(filepath, "%s/%s", "/data/data/com.sony.openclexample1/app_execdir/binariesKernel/", next_file->d_name);
////	        remove(filepath);
////	    }
//
//	string dirtemp="/data/data/com.sony.openclexample1/app_execdir/configToPrecompile.txt";
//	char * st=ReadFile((char*)dirtemp.c_str());
////
////	dirtemp="/data/data/com.sony.openclexample1/app_execdir/olivierdata/test4/manifest4bis.txt"
//	LOGI("%s",st);
//
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir (".")) != NULL) {
	  /* print all the files and directories within directory */
	  while ((ent = readdir (dir)) != NULL) {
//		  if ((ent->d_name!="..")&&(ent->d_name!="."))
//		  	  remove( "myfile.txt" );
	  LOGI ("%s\n", ent->d_name);
	  }
	  closedir (dir);
	} else {
	  /* could not open directory */
	  perror ("");
	  return EXIT_FAILURE;
	}
//if (0){
	//if (1){

	//t->trainCmd("./train numepochs=5 netdef=8c5z-relu-mp2-16c5z-relu-mp3-150n-tanh-10n learningrate=0.002 dataset=mnist");///data/data/com.sony.openclexample1/app_execdir
	//t->trainCmd("./train numtest=-1 numtrain=10000 datadir=/sdcard1/olivierdata");
//8c5z-relu-mp2-16c5z-relu-mp3-

	//1s8c5z-relu-mp2-1s16c5z-relu-mp3-
//-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n

//7janv t->trainCmd("./train datadir=/data/data/com.sony.openclexample1/app_execdir/test4/ netdef=1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n numepochs=3 batchsize=128 trainfile=manifest.txt validatefile=manifest.txt numtrain=2048  numtest=2048");
//1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
		LOGI("###############");
	TrainModel* t= new TrainModel();

//	string filename_label="/data/data/com.sony.openclexample1/memMapFileLabel60000MNIST.raw";
//			string filename_data="/data/data/com.sony.openclexample1/memMapFileData60000MNIST.raw";
//			int imageSize=32;
//			int numOfChannel=3;//black and white => 1; color =>3
//			string storeweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstTransferedTEST.dat";
//			string loadweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";
//			string networkDefinition="1s32c5z-mp2-1s32c5z-mp2-1s64c5z-mp2-64n-relu-10n";
//			string validationSet="t10k-images-idx3-ubyte";
//			int numepochs=100;
//			int batchsize=128;
//			int numtrain=128;//59904;//
//			int numtest=128;//9984;//
//			float learningRate=0.0001f;
//
//			string cmdString="train filename_label="+filename_label;
//			cmdString=cmdString+" filename_data="+filename_data;
//			cmdString=cmdString+" imageSize="+to_string(imageSize);
//			cmdString=cmdString+" numPlanes="+to_string(numOfChannel);
//			cmdString=cmdString+" storeweightsfile="+storeweightsfile;
//			cmdString=cmdString+" loadweightsfile="+loadweightsfile;
//			cmdString=cmdString+" netdef="+networkDefinition;
//			cmdString=cmdString+" numepochs="+to_string(numepochs);
//			cmdString=cmdString+" batchsize="+to_string(batchsize);
//			cmdString=cmdString+" numtrain="+to_string(numtrain);
//			cmdString=cmdString+" numtest="+to_string(numtest);
//			cmdString=cmdString+" validatefile="+validationSet;
//			cmdString=cmdString+" learningrate="+to_string(learningRate);
	#if TRANSFER ==1
		string filename_label="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw";
		string filename_data="/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw";
		int imageSize=28;
		int numOfChannel=1;//black and white => 1; color =>3
		string storeweightsfile="/data/data/com.sony.openclexample1/directoryTest/weightstTransferedTEST.dat";
		string loadweightsfile="/data/data/com.sony.openclexample1/directoryTest/weightstface1.dat";
		string loadnormalizationfile="/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt";
		string networkDefinition="1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n";
		int numepochs=100;
		int batchsize=128;
		int numtrain=128;
		float learningRate=0.01f;

		string cmdString="train filename_label="+filename_label;
		cmdString=cmdString+" filename_data="+filename_data;
		cmdString=cmdString+" imageSize="+to_string(imageSize);
		cmdString=cmdString+" numPlanes="+to_string(numOfChannel);
		cmdString=cmdString+" storeweightsfile="+storeweightsfile;
		cmdString=cmdString+" loadweightsfile="+loadweightsfile;
		cmdString=cmdString+" loadnormalizationfile="+loadnormalizationfile;
		cmdString=cmdString+" netdef="+networkDefinition;
		cmdString=cmdString+" numepochs="+to_string(numepochs);
		cmdString=cmdString+" batchsize="+to_string(batchsize);
		cmdString=cmdString+" numtrain="+to_string(numtrain);
		cmdString=cmdString+" learningrate="+to_string(learningRate);

	#endif
	#if TRANSFER ==0
		string filename_label="/data/data/com.sony.openclexample1/memMapFileLabel60000MNIST.raw";
		string filename_data="/data/data/com.sony.openclexample1/memMapFileData60000MNIST.raw";
		int imageSize=28;
		int numOfChannel=1;//black and white => 1; color =>3
		string storeweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstTransferedTEST.dat";
		string loadweightsfile="/data/data/com.sony.openclexample1/preloadingData/weightstface1.dat";
		string loadnormalizationfile="/data/data/com.sony.openclexample1/preloadingData/normalization.txt";
		string networkDefinition="1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n";
		string validationSet="t10k-images-idx3-ubyte";
		int numepochs=3;
		int batchsize=100;//128;
		int numtrain=60000;//59904;//
		int numtest=10000;//9984;//
		float learningRate=0.0001f;

		string cmdString="train filename_label="+filename_label;
		cmdString=cmdString+" filename_data="+filename_data;
		cmdString=cmdString+" imageSize="+to_string(imageSize);
		cmdString=cmdString+" numPlanes="+to_string(numOfChannel);
		cmdString=cmdString+" storeweightsfile="+storeweightsfile;
		cmdString=cmdString+" loadweightsfile="+loadweightsfile;
		cmdString=cmdString+" loadnormalizationfile="+loadnormalizationfile;
		cmdString=cmdString+" netdef="+networkDefinition;
		cmdString=cmdString+" numepochs="+to_string(numepochs);
		cmdString=cmdString+" batchsize="+to_string(batchsize);
		cmdString=cmdString+" numtrain="+to_string(numtrain);
		cmdString=cmdString+" numtest="+to_string(numtest);
		cmdString=cmdString+" validatefile="+validationSet;
		cmdString=cmdString+" learningrate="+to_string(learningRate);

	#endif
		LOGI("cmd %s",cmdString.c_str());
		t->trainCmd(cmdString);

	delete t;

	clock_gettime(CLOCK_MONOTONIC, &end1);
	elapsedSeconds1 = TimeSpecToSeconds(&end1) - TimeSpecToSeconds(&start1);
	LOGI("3)time %f\n\n",elapsedSeconds1);
	/*get the end time*/
	gettimeofday(&end, NULL);
	/*Print the amount of time taken to execute*/
	LOGI("%f\n ms", (float)(((end.tv_sec * 1000000 + end.tv_usec)	- (start.tv_sec * 1000000 + start.tv_usec))/1000));


        return 0;
}



extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_runNativeC(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{


	PredictionModel* p=new PredictionModel();
	//p->predictCmd("./predict batchsize=128 inputfile=/sdcard1/testdata4/manifest2.txt outputfile=/data/data/com.sony.openclexample1/app_execdir/pred2.txt");
	p->predictCmd("./predict weightsfile=/data/data/com.sony.openclexample1/preloadingData/weightstTransferedTEST.dat  inputfile=/sdcard1/character/manifest4.txt outputfile=/data/data/com.sony.openclexample1/preloadingData/pred2.txt");
	delete p;


	return 0;
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_prepareFiles(JNIEnv* env, jclass clazz, jobject bitmapIn, jobject bitmapOut, jintArray info)
{
	TrainModel* t= new TrainModel();
	t->prepareFiles("/sdcard1/character/manifest6.txt",128, 1,"/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw","/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw","/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt");

	delete t;


	return 0;
}

