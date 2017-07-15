
#include <stdio.h>
#include "sonyOpenCLexample1.h"

#include <android/bitmap.h>
#include <jni.h>


#include "trainEngine/train.h"
#include "predictEngine/predict.h"



//#include "kernelManager/kernelManager.h"

#include <sys/stat.h>
#include <dirent.h>

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

////#include "train.h"
//#define CLMATH_VERBOSE 1
//#define DEEPCL_VERBOSE 1


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

std::string jstring2string(JNIEnv *env, jstring jStr) {
    if (!jStr)
        return "";

    std::vector<char> charsCode;
    const jchar *chars = env->GetStringChars(jStr, NULL);
    jsize len = env->GetStringLength(jStr);
    jsize i;

    for( i=0;i<len; i++){
        int code = (int)chars[i];
        charsCode.push_back( code );
    }

    env->ReleaseStringChars(jStr, chars);
    return std::string(charsCode.begin(), charsCode.end());
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_training(JNIEnv* env, jclass clazz, jstring javaString)
{



	 struct timespec start1;
		struct timespec end1;
		double elapsedSeconds1;
		clock_gettime(CLOCK_MONOTONIC, &start1);

		string path="";
		path=path+jstring2string(env,javaString);
		string path2=path+"directoryTest/";

		struct timeval start, end;
		/*get the start time*/
		gettimeofday(&start, NULL);


		LOGI("###############");
	TrainModel* t= new TrainModel(path2);

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
		t->trainCmd(cmdString,path2);

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
Java_com_sony_openclexample1_OpenCLActivity_prediction(JNIEnv* env, jclass clazz, jstring javaString)
{

	string path="";
	path=path+jstring2string(env,javaString);
	string path2=path+"directoryTest/";
	PredictionModel* p=new PredictionModel(path2);
	p->predictCmd("./predict weightsfile=/data/data/com.sony.openclexample1/preloadingData/weightstTransferedTEST.dat  inputfile=/sdcard1/character/manifest4.txt outputfile=/data/data/com.sony.openclexample1/preloadingData/pred2.txt");
	delete p;


	return 0;
}

extern "C" jint
Java_com_sony_openclexample1_OpenCLActivity_prepareFiles(JNIEnv* env, jclass clazz, jstring javaString)
{
	string path="";
	path=path+jstring2string(env,javaString);
	int mkdirretval;
	//create path
	string path2=path+"directoryTest/";
	mkdirretval=mkpath(path2.c_str(),0755);
	if (-1 == mkdirretval)
	{
		LOGI("Error creating directory!n");
	}
	LOGI("1");

	string path3=path+"directoryTest/binariesKernel/";
	mkdirretval=mkpath(path3.c_str(),0755);
	if (-1 == mkdirretval)
	{
		LOGI("Error creating directory!n");
	}
	LOGI("2");
	TrainModel* t= new TrainModel(path2);
	t->prepareFiles("/sdcard1/character/manifest6.txt",128, 1,"/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw","/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw","/data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt");

	delete t;


	return 0;
}

