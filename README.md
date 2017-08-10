TransferCL
======== 

Table of Contents
=================

* [1. Why TransferCL ?](#1-why-transfercl-)
* [2. How does it work?](#2-how-does-it-work)
	* [2.1 Transfer Learning](#21-transfer-learning)
* [3. Installation](#3-installation)
	* [3.1 Installation from prebuild packages](#31-installation-from-prebuild-packages)
	* [3.2 Building from source: Native Library installation](#32-building-from-source-native-library-installation)
	   * [3.1.1 Pre-requisites](#311-pre-requisites)
		  * [3.1.1.1 Where to find the appropriated OpenCL shared-library](#3111-where-to-find-the-appropriated-opencl-shared-library)
	   * [3.1.2 Procedure](#312-procedure)
	   * [3.1.3 Android application installation](#313-android-application-installation)
* [4. How to use it](#4-how-to-use-it)
* [5. Case study](#5-case-study)
* [6. Important remarks](#6-important-remarks)
* [7. How to see the output](#7-how-to-see-the-output)
* [8. To get in contact](#8-to-get-in-contact)
* [9. Contribute](#9-contribute)



TransferCL is an open source deep learning framework which has been designed to run on mobile devices.  The goal is to enable mobile devices to tackle complex deep learning oriented-problem reserved to desktop computers. This project has been initiated by the parallel and distributed processing laboratory at [National Taiwan University](https://www.ntu.edu.tw).  Olivier Valery develloped this tool during his PhD at National Taiwan University. TransferCL is released under Mozilla Public Licence 2.0.

### 1. Why TransferCL ?

Recent mobile devices are equipped with multiple sensors, which can give insight into the mobile users' profile.  We believe that such information can be used to customize the mobile experience for a specific user.

The primary goal of TransferCL is to leverage Transfer Learning on mobile devices. Our work is based on the [DeepCL Library](https://github.com/hughperkins/DeepCL). Despite their similarities, TransferCL has been designed to run efficiently on a broad range of mobile devices. As a result, TransferCL implements its own memory management system and own OpenCL kernels in order take into account the specificity of mobile devices' System-on-Chip.

### 2. How does it work?

TransferCL is implemented in C++ and is able to run on any Android device with an OpenCL compliant GPU (the vast majority of modern Android devices). TransferCL provides several APIs which allow programmers to transparently leverage deep learning on mobile devices.

#### 2.1 Transfer Learning

Modern mobile devices suffer from two major issues that prevent them from training a deep neural network on mobile devices via a classic supervised learning approach:

* First, the computing capabilities are relatively limited in comparison to servers.
* Then, a single mobile device may not have a sufficient label data in its training data set to train a deep neural network accurately.

![file architecture](/image/traditional_ml_setup.png?raw=true)

Transfer learning is a technique that shortcuts a lot of this work by taking a fully-trained model for a set of categories like ImageNet and retraining from the existing weights for new classes.  The use of pre-trained features is currently the most straightforward and most commonly way to perform transfer learning, but it is not the only one.

![file architecture](/image/transfer_learning_setup.png?raw=true)

For more information, please check these websites:
* [Transfer Learning with TensorFlow](https://www.tensorflow.org/tutorials/image_retraining)
* [Transfer Learning with CNTK](https://docs.microsoft.com/en-us/cognitive-toolkit/Build-your-own-image-classifier-using-Transfer-Learning)
* [A survey of existing Transfer Learning techniques](http://ruder.io/transfer-learning/index.html)
* [The class Convolutional Neural Networks for Visual Recognition at Stanford ](http://cs231n.github.io/transfer-learning/)
* [A paper study of transfer learning performance](https://arxiv.org/abs/1411.1792)

### 3. Installation

There are two ways to install TranferCL: 
1. From the source 
	* This method enables the developer to build TranferCL for any particular mobile device architecture. We recommend this approach.
2. Importing TranferCL from our prebuilt directory
	* TransferCL has been pre-build for several commonly used hardware configurations. For these configurations, the shared-library can be imported directly in the Android application. However, we emphasize that once built, a shared-library is specific to a CPU ABI (armeabi-v7a, arm64-v8a ...) a GPU architecture (Adreno, Mali ...) and won't work for any other configurations than the one targeted initially.

#### 3.1 Installation from prebuild packages

* In the folder ```prebuild library```, you can find the binary files (to include in your Android aplication) and the JavaWrapprer.
* In this folder, this [file](prebuild%20library/README.md) includes more details about their utilization.
	
#### 3.2 Building from source: Native Library installation

##### 3.1.1 Pre-requisites

* OpenCL compliant GPU, along with appropriate OpenCL driver:
    * The ```libOpenCL.so```, corresponding to the mobile device's GPU which is being targeting, need to to be placed in the folder ```extra_libs```.
    * the headers files (*.h) need to be placed in the folder ```include``` 
    
* CrystaX NDK: 
    * [Google NDK](https://developer.android.com/ndk/index.html) provides a set of tools to build native applications on Android.  Our work is based on [CrystaX NDK](https://www.crystax.net/en), which has been developed as a drop-in replacement for Google NDK. For more information, please check their [website](https://www.crystax.net/en).
    * It is still possible to use Google NDK, however, the user will need the import ```Boost C++``` by himself.
	
###### 3.1.1.1 Where to find the appropriated OpenCL shared-library

As mentioned previously, the installation of TransferCL requires the compatible ```libOpenCL.so``` library and the corresponding OpenCL headers:
* The headers: the simplest way is extracting them from Adreno/Mali SDK. For Adreno SDK, they can be found at ```<Adreno_SDK>/Development/Inc/CL```. For Mali SDK, they can be found at ```<MALI_SDK>/include/CL```.
* The ```libOpenCL.so```:  the library is generally already present on the mobile device and can be pulled via ```adb pull /system/vendor/lib/libOpenCL.so .```. (the path may change from one brand to another)


##### 3.1.2 Procedure

* git clone https://github.com/OValery16/TransferCL.git
* add your libOpenCL.so in the folder ```extra_libs```.
* add the OpenCL header in the folder ```include```.

Your repository should look like that:

![file architecture](/image/files2.png?raw=true)

* In the folder 'jni', create a ```\*.cpp``` file and a ```&ast.h``` file, whose role is to interface with TranferCL. The Android application will call this file's method to interact with the deep learning network.
    * An example can be found in ```transferCLinterface.cpp```
    * The name of the functions need to be modified in order to respect the naming convention for native function in NDK/JNI application: ```Java_{package_and_classname}_{function_name}(JNI arguments)```
        * For example the ```Java_com_sony_openclexample1_OpenCLActivity_training``` means that this method is mapped to the ```training``` method from the  ```OpenCLActivity``` activity in the ```com.sony.openclexample1``` package.
        * For more information about this naming convention, please check this [website](https://www3.ntu.edu.sg/home/ehchua/programming/java/JavaNativeInterface.html)
* In the 'Android.mk', change the line ```LOCAL_SRC_FILES :=transferCLinterface.cpp``` to ```LOCAL_SRC_FILES :={your_file_name}.cpp``` (replace 'your_file_name' by the name of the file you just created)
* In the 'Application.mk' change the line ```APP_ABI := armeabi-v7a``` to ```APP_ABI := {the_ABI_you_want_to_target}``` (replace 'the_ABI_you_want_to_target' by the ABI you want to target)
    * A list of all supported ABIs is given on the [NDK website](https://developer.android.com/ndk/guides/abis.html).
    * Make sure that your device supports the chosen ABI (otherwise it won't be able to find TransferCL 's methods). If you are not certain, you can check which ABIs are supported by your device, via some android applications like ```OpenCL-Z```.
* Run CrystaX  NDK to build your shared library with the command ```ndk-build``` (crystax-ndk-X\ndk-build where X is CrystaX NDK version)
```
>ndk-build
```
* CrystaX NDK will output several shared library files (they are specific to your mobile device ABIs)

##### 3.1.3 Android application installation

* Create your Android project.
* Don't forget to respect the name conversion that you chose earlier (otherwise your application won't find your native methods)
* In your activity, you have to load your native library as following
```Java
    static {
      try {
          System.loadLibrary("openclexample1");  //Just put your libaries name
      }
      catch (UnsatisfiedLinkError e) {
          sfoundLibrary = false;
      }
    }
```
* Define the methods that have been implemented natively (in the shared library) as in the example
```
    //the name need to be the same as the one defined in the shared library
    public static native int training(String path, String cmdTrain);
    public static native int prediction(String path, String cmdPrediction);
    public static native int prepareFiles(String path, String fileNameStoreData,String fileNameStoreLabel, String fileNameStoreNormalization, String manifestPath, int nbImage, int imagesChannelNb);
```
* Build your applications

## 4. How to use it

* In the folder ```study case```, you can find a template application  using TranferCL. This application defines 2 Java source package:
	* ```com.transferCL```, which is a java wrapper for the native methods defined in TranferCL (```TransferCLlib.java```).
	* ```com.example.myapplication```, which is an android activity (```MainActivity.java```). It calls  the methods declared in ```TransferCLlib.java```.
* In the folder ```prebuild library```, you can also find the java wrapper file (```TransferCLlib.java```).
* In the file ```TransferCLlib.java``` , you can find three methods that have been already defined:
    * ```prepareFiles(String path, String fileNameStoreData,String fileNameStoreLabel, String fileNameStoreNormalization, String manifestPath, int nbImage, int imagesChannelNb)```
        * This method builds the training data set.
        * Originally the training data set is stored on the microSD card as a set of jpeg images and a manifest file as defined on [DeepCL website in the section 'jpegs'](https://github.com/hughperkins/DeepCL/blob/master/doc/Loaders.md)
            * In future versions of this tutorial there will be some concrete examples.
        * The images are processed by TransferCL and stored on the mobile device as a unique binary file.
        * Also create the folder architecture on your mobile device to store pre-build OpenCL kernel.
            * If these folders are not created, the application will crash.
        * This method has to be the first to run.	
    * ```training(String path, String cmdTrain);```
        * This method trains the new deep neural network. 
        * This method reuse the previously created files.
        * This method also build the OpenCL kernel the system need to train the deep neural network. 
        * The parameters of the training methods are given in 'transferCLinterface.cpp'
    * ```prediction(String path, String cmdPrediction)```
        * This method performs the inference task and store the result in a text file

* Currently the most convenient way is to use [DeepCL Library](https://github.com/hughperkins/DeepCL) to train the first deep learning model on mobile.
    * However a conversion tool (TensorFlow model/TransferCL) is in development.

## 5. Case study


* A case study is in the folder ```case study```
* In the folder ```study case```, you can find an application template using TranferCL.
* You can also find a [tutorial](./case%20study/README.md).
* The study case explores the following scenario:	
	* Training on a server
		1. We train a network (LeNet5) on the server with MNIST dataset (the training configuration is the standard one).
		2. The final model is stored on the server in a binary file. 
		3. This binary file is copied on the mobile device (for example, on the SD Card).
	* Files preparations (```prepareFiles```)
		1. We create the working directory ```directoryTest``` (perform at the native level by TransferCL)
		2. The training files (the training file and their labels are respectively stored in one binary file) are generated.
		3. TransferCL analyse the dataset, stores its mean/stdDev and store them in one file
	* Training on the mobile device (```training```)
		1. TransferCL creates a neural network, and initializes the weights of all layers except the last one with the weights of the pre-trained network. 
		2. The last layer is initialized with a random number generator.
		3. The training starts: TransferCL train the final layer from scratch, while leaving all the others untouched.
			1. TransferCL performs the forward propagation.
			2. TransferCL performs the backward propagation and the weight update only on the last layer.
		4. After very few iterations, the prediction error drops significantly. Most images' label are predicted correctly after only 20 iterations. (```loss=98.937355 numRight=118```)
	* Test on the mobile device (```prediction```)
		1. We tested our model prediction accuracy with a test dataset, which our model has never seen. In our expleriment, TransferCL predicted all test images label correctly.
* To Conclude this case study, TransferCL trained on only about 12 images per class (a total of 10 classes) in a few seconds and predicted all test images correctly.

## 6. Important remarks

* The training images must cover sufficiently the scenarios that you want to predict. If the classifier sees fully new concepts or contexts, it is likely to perform badly. It applies in particular when leveraging transfer learning in a mobile device environment.

    * If the training dataset only contains images from a constraint environment (say, indoor), the classifier won't be able to score images accurately from a different environment (outdoor).
    * If the test images have widely different characteristics (illumination, background, color, size, position, etc), the classifier won't be able to perform very well.
    * If a test image contains entirely new concepts, the classifier won't be able to identify its class.
    
* The choice of the base model to transfer to the mobile device is very important. The two classification tasks (the one on the server and the one on the mobile device) should be related. For example, in our case study the base network has been trained to recognize handwritten digits and this knowledge is transferred to TransferCL in order to train a new network to classify handwritten characters on mobile devices.

## 7. How to see the output 

* In order to see the ouput of TranferCL, you can use the [logcat command-line tool](https://developer.android.com/studio/command-line/logcat.html):
	* ```>adb logcat ActivityManager:I TransferCL:D *:S```

* For example, the output of ```prepareFiles(String path, String fileNameStoreData,String fileNameStoreLabel, String fileNameStoreNormalization, String manifestPath, int nbImage, int imagesChannelNb)``` should look like that.

```
I/TransferCL(10924): -------Files Preparation
I/TransferCL(10924): -----------Generation of the memory-map files (binary files)
I/TransferCL(10924): -------------- 128 images in the training set with 1 dimension and a image size of 28 X 28
I/TransferCL(10924): -------------- Training set loading
I/TransferCL(10924): -------------- training data file generation: /data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw
I/TransferCL(10924): -------------- label file generation: /data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw
I/TransferCL(10924): -------------- normalization file file generation: /data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt
I/TransferCL(10924): -------File generation: completed
I/TransferCL(10924): easyCL oject destroyed
```

* For example, the output of ```training(String path, String cmdTrain);``` should look like that.
	* After each iteration, TransferCL displays the loss value and the number of images' label correctly predicted. 
		* For example, ```loss=98.937355 numRight=118``` means that the loss is equal to 98.937355 and 118 images' label have been correctly predicted (we have only 128 images in the training dataset)
		
```
I/TransferCL(10924): ################################################
I/TransferCL(10924): ###################Training#####################
I/TransferCL(10924): ################################################
I/TransferCL(10924): ------- Loading configuration: training set 128 images with 1 channel and an image size of 28 X 28
I/TransferCL(10924): ------- Network Generation: 1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
I/TransferCL(10924): -----------Define Weights Initialization Method
I/TransferCL(10924): -------------- Chosen Initializer: original
I/TransferCL(10924): -----------Network Layer Generation
I/TransferCL(10924): -----------Selecting Training method (sgd)
I/TransferCL(10924): -----------Loading the weights (the othe weights are radomly initialized with the initializer defined previously)
I/TransferCL(10924): -----------Set up Trainer
I/TransferCL(10924):
I/TransferCL(10924):
I/TransferCL(10924): ################################################
I/TransferCL(10924): ################Start learning##################
I/TransferCL(10924): ################################################
I/TransferCL(10924):
I/TransferCL(10924):
I/TransferCL(10924): loss=340.038452 numRight=16
I/TransferCL(10924): loss=275.497894 numRight=28
I/TransferCL(10924): loss=250.139160 numRight=45
I/TransferCL(10924): loss=231.197174 numRight=60
I/TransferCL(10924): loss=215.234680 numRight=71
I/TransferCL(10924): loss=201.223007 numRight=80
I/TransferCL(10924): loss=188.762817 numRight=87
I/TransferCL(10924): loss=177.619720 numRight=92
I/TransferCL(10924): loss=167.614990 numRight=97
I/TransferCL(10924): loss=158.600357 numRight=105
I/TransferCL(10924): loss=150.450089 numRight=108
I/TransferCL(10924): loss=143.057266 numRight=109
I/TransferCL(10924): loss=136.329941 numRight=111
I/TransferCL(10924): loss=130.189621 numRight=113
I/TransferCL(10924): loss=124.568176 numRight=114
I/TransferCL(10924): loss=119.406982 numRight=114
I/TransferCL(10924): loss=114.655197 numRight=116
I/TransferCL(10924): loss=110.268494 numRight=117
I/TransferCL(10924): loss=106.208397 numRight=118
I/TransferCL(10924): loss=102.441200 numRight=118
I/TransferCL(10924): loss=98.937355 numRight=118
I/TransferCL(10924): loss=95.670959 numRight=118
I/TransferCL(10924): loss=92.619194 numRight=119
I/TransferCL(10924): loss=89.761833 numRight=119
I/TransferCL(10924): loss=87.081139 numRight=119
I/TransferCL(10924): loss=84.561211 numRight=119
I/TransferCL(10924): loss=82.188004 numRight=119
I/TransferCL(10924): loss=79.949051 numRight=119
I/TransferCL(10924): loss=77.833115 numRight=119
I/TransferCL(10924): loss=75.830170 numRight=120
I/TransferCL(10924): loss=73.931221 numRight=121
I/TransferCL(10924): loss=72.128242 numRight=121
I/TransferCL(10924): loss=70.413902 numRight=122
I/TransferCL(10924): loss=68.781647 numRight=122
I/TransferCL(10924): loss=67.225548 numRight=122
I/TransferCL(10924): loss=65.740196 numRight=122
I/TransferCL(10924): loss=64.320671 numRight=122
I/TransferCL(10924): loss=62.962559 numRight=122
I/TransferCL(10924): loss=61.661789 numRight=122
I/TransferCL(10924): loss=60.414635 numRight=122
I/TransferCL(10924): loss=59.217720 numRight=123
I/TransferCL(10924): loss=58.067921 numRight=123
I/TransferCL(10924): loss=56.962376 numRight=123
I/TransferCL(10924): loss=55.898487 numRight=123
I/TransferCL(10924): loss=54.873783 numRight=123
I/TransferCL(10924): loss=53.886082 numRight=123
I/TransferCL(10924): loss=52.933289 numRight=123
I/TransferCL(10924): loss=52.013527 numRight=123
I/TransferCL(10924): loss=51.125000 numRight=123
I/TransferCL(10924): loss=50.266037 numRight=123
I/TransferCL(10924): loss=49.435169 numRight=123
I/TransferCL(10924): loss=48.630978 numRight=123
I/TransferCL(10924): loss=47.852093 numRight=124
I/TransferCL(10924): loss=47.097301 numRight=124
I/TransferCL(10924): loss=46.365482 numRight=124
I/TransferCL(10924): loss=45.655514 numRight=124
I/TransferCL(10924): loss=44.966412 numRight=124
I/TransferCL(10924): loss=44.297222 numRight=124
I/TransferCL(10924): loss=43.647057 numRight=124
I/TransferCL(10924): loss=43.015118 numRight=124
I/TransferCL(10924): loss=42.400551 numRight=124
I/TransferCL(10924): loss=41.802689 numRight=124
I/TransferCL(10924): loss=41.220783 numRight=124
I/TransferCL(10924): loss=40.654228 numRight=124
I/TransferCL(10924): loss=40.102375 numRight=124
I/TransferCL(10924): loss=39.564655 numRight=124
I/TransferCL(10924): loss=39.040504 numRight=124
I/TransferCL(10924): loss=38.529396 numRight=124
I/TransferCL(10924): loss=38.030872 numRight=124
I/TransferCL(10924): loss=37.544403 numRight=124
I/TransferCL(10924): loss=37.069607 numRight=124
I/TransferCL(10924): loss=36.606026 numRight=124
I/TransferCL(10924): loss=36.153259 numRight=124
I/TransferCL(10924): loss=35.710938 numRight=124
I/TransferCL(10924): loss=35.278702 numRight=124
I/TransferCL(10924): loss=34.856178 numRight=124
I/TransferCL(10924): loss=34.443069 numRight=124
I/TransferCL(10924): loss=34.039047 numRight=124
I/TransferCL(10924): loss=33.643822 numRight=125
I/TransferCL(10924): loss=33.257088 numRight=125
I/TransferCL(10924): loss=32.878590 numRight=125
I/TransferCL(10924): loss=32.508057 numRight=125
I/TransferCL(10924): loss=32.145245 numRight=125
I/TransferCL(10924): loss=31.789917 numRight=125
I/TransferCL(10924): loss=31.441833 numRight=125
I/TransferCL(10924): loss=31.100786 numRight=125
I/TransferCL(10924): loss=30.766573 numRight=125
I/TransferCL(10924): loss=30.438965 numRight=125
I/TransferCL(10924): loss=30.117785 numRight=125
I/TransferCL(10924): loss=29.802849 numRight=125
I/TransferCL(10924): loss=29.493982 numRight=125
I/TransferCL(10924): loss=29.191004 numRight=125
I/TransferCL(10924): loss=28.893751 numRight=125
I/TransferCL(10924): loss=28.602062 numRight=125
I/TransferCL(10924): loss=28.315805 numRight=125
I/TransferCL(10924): loss=28.034807 numRight=125
I/TransferCL(10924): loss=27.758936 numRight=125
I/TransferCL(10924): loss=27.488062 numRight=125
I/TransferCL(10924): loss=27.222054 numRight=125
I/TransferCL(10924): loss=26.960766 numRight=126
I/TransferCL(10924): loss=26.704094 numRight=126
I/TransferCL(10924): loss=26.451908 numRight=126
I/TransferCL(10924): loss=26.204117 numRight=127
I/TransferCL(10924): loss=25.960588 numRight=127
I/TransferCL(10924): loss=25.721230 numRight=128
I/TransferCL(10924): loss=25.485914 numRight=128
I/TransferCL(10924): loss=25.254572 numRight=128
I/TransferCL(10924): loss=25.027102 numRight=128
I/TransferCL(10924): loss=24.803392 numRight=128
I/TransferCL(10924): loss=24.583364 numRight=128
I/TransferCL(10924): loss=24.366938 numRight=128
I/TransferCL(10924): loss=24.154015 numRight=128
I/TransferCL(10924): loss=23.944517 numRight=128
I/TransferCL(10924): loss=23.738382 numRight=128
I/TransferCL(10924): loss=23.535511 numRight=128
I/TransferCL(10924): loss=23.335838 numRight=128
I/TransferCL(10924): loss=23.139290 numRight=128
I/TransferCL(10924): loss=22.945807 numRight=128
I/TransferCL(10924): loss=22.755301 numRight=128
I/TransferCL(10924): loss=22.567717 numRight=128
I/TransferCL(10924): loss=22.382999 numRight=128
I/TransferCL(10924): loss=22.201069 numRight=128
I/TransferCL(10924): loss=22.021870 numRight=128
I/TransferCL(10924): loss=21.845362 numRight=128
I/TransferCL(10924): loss=21.671467 numRight=128
I/TransferCL(10924): loss=21.500137 numRight=128
I/TransferCL(10924): loss=21.331320 numRight=128
I/TransferCL(10924): loss=21.164948 numRight=128
I/TransferCL(10924): loss=21.000994 numRight=128
I/TransferCL(10924): loss=20.839394 numRight=128
I/TransferCL(10924): loss=20.680103 numRight=128
I/TransferCL(10924): loss=20.523073 numRight=128
I/TransferCL(10924): loss=20.368263 numRight=128
I/TransferCL(10924): loss=20.215612 numRight=128
I/TransferCL(10924): loss=20.065100 numRight=128
I/TransferCL(10924): loss=19.916662 numRight=128
I/TransferCL(10924): loss=19.770279 numRight=128
I/TransferCL(10924): loss=19.625898 numRight=128
I/TransferCL(10924): loss=19.483482 numRight=128
I/TransferCL(10924): loss=19.342997 numRight=128
I/TransferCL(10924): loss=19.204390 numRight=128
I/TransferCL(10924): loss=19.067646 numRight=128
I/TransferCL(10924): loss=18.932722 numRight=128
I/TransferCL(10924): loss=18.799574 numRight=128
I/TransferCL(10924): loss=18.668186 numRight=128
I/TransferCL(10924): loss=18.538515 numRight=128
I/TransferCL(10924): loss=18.410522 numRight=128
I/TransferCL(10924): loss=18.284184 numRight=128
I/TransferCL(10924): loss=18.159477 numRight=128
I/TransferCL(10924): loss=18.036362 numRight=128
I/TransferCL(10924): loss=17.914803 numRight=128
I/TransferCL(10924): loss=17.794794 numRight=128
I/TransferCL(10924): loss=17.676283 numRight=128
I/TransferCL(10924): loss=17.559261 numRight=128
I/TransferCL(10924): loss=17.443693 numRight=128
I/TransferCL(10924): loss=17.329556 numRight=128
I/TransferCL(10924): loss=17.216822 numRight=128
I/TransferCL(10924): loss=17.105467 numRight=128
I/TransferCL(10924): loss=16.995462 numRight=128
I/TransferCL(10924): loss=16.886803 numRight=128
I/TransferCL(10924): loss=16.779446 numRight=128
I/TransferCL(10924): loss=16.673370 numRight=128
I/TransferCL(10924): loss=16.568565 numRight=128
I/TransferCL(10924): loss=16.465002 numRight=128
I/TransferCL(10924): loss=16.362663 numRight=128
I/TransferCL(10924): loss=16.261522 numRight=128
I/TransferCL(10924): loss=16.161564 numRight=128
I/TransferCL(10924): loss=16.062769 numRight=128
I/TransferCL(10924): loss=15.965113 numRight=128
I/TransferCL(10924): loss=15.868578 numRight=128
I/TransferCL(10924): loss=15.773150 numRight=128
I/TransferCL(10924): loss=15.678812 numRight=128
I/TransferCL(10924): loss=15.585545 numRight=128
I/TransferCL(10924): loss=15.493321 numRight=128
I/TransferCL(10924): loss=15.402135 numRight=128
I/TransferCL(10924): loss=15.311967 numRight=128
I/TransferCL(10924): loss=15.222803 numRight=128
I/TransferCL(10924): loss=15.134621 numRight=128
I/TransferCL(10924): loss=15.047411 numRight=128
I/TransferCL(10924): loss=14.961153 numRight=128
I/TransferCL(10924): loss=14.875841 numRight=128
I/TransferCL(10924): loss=14.791453 numRight=128
I/TransferCL(10924): loss=14.707974 numRight=128
I/TransferCL(10924): loss=14.625394 numRight=128
I/TransferCL(10924): loss=14.543686 numRight=128
I/TransferCL(10924): loss=14.462858 numRight=128
I/TransferCL(10924): loss=14.382881 numRight=128
I/TransferCL(10924): loss=14.303750 numRight=128
I/TransferCL(10924): loss=14.225451 numRight=128
I/TransferCL(10924): loss=14.147968 numRight=128
I/TransferCL(10924): loss=14.071287 numRight=128
I/TransferCL(10924): loss=13.995400 numRight=128
I/TransferCL(10924): loss=13.920297 numRight=128
I/TransferCL(10924): loss=13.845965 numRight=128
I/TransferCL(10924): loss=13.772382 numRight=128
I/TransferCL(10924): loss=13.699554 numRight=128
I/TransferCL(10924): loss=13.627465 numRight=128
I/TransferCL(10924): loss=13.556102 numRight=128
I/TransferCL(10924): loss=13.485450 numRight=128
I/TransferCL(10924): loss=13.415505 numRight=128
I/TransferCL(10924): gettimeofday 8387.000000
I/TransferCL(10924):  ms
I/TransferCL(10924): -----------End of ther training: Delete object
I/TransferCL(10924): -----------Delete weightsInitializer
I/TransferCL(10924): -----------Delete trainer
I/TransferCL(10924): -----------Delete netLearner
I/TransferCL(10924): -----------Delete net
I/TransferCL(10924): -----------Delete trainLoader
I/TransferCL(10924): All code took 8451.000000
I/TransferCL(10924):  ms
I/TransferCL(10924): easyCL oject destroyed
I/TransferCL(10924): 3)time 8.484612
I/TransferCL(10924):
I/TransferCL(10924): 8484.000000
I/TransferCL(10924):  ms
```

* For example, the output of ```prediction(String path, String cmdPrediction);``` should look like that.

```
I/TransferCL(14885): ################################################
I/TransferCL(14885): ###################Prediction###################
I/TransferCL(14885): ################################################
I/TransferCL(14885): ------- Network Generation
I/TransferCL(14885): -----------Network Layers Creation 1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
I/TransferCL(14885): -----------Loading the weights
I/TransferCL(14885): -----------Start prediction
I/TransferCL(14885): --------- Prediction: done (prediction in /data/data/com.sony.openclexample1/preloadingData/pred2.txt)
I/TransferCL(14885): --------- End of ther prediction: Delete objects
I/TransferCL(14885): easyCL oject destroyed
```

## 8. To get in contact

Just create issues (in GitHub) in the top right of this page. Don't worry about whether you think your issue sounds unimportant or trivial. The more feedback we can get, the better!

## 9. Contribute

If you are interestered in this project, feel free to contact me.



