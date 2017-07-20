## TransferCL

TransferCL is an open source deep learning framework which has been designed to run on mobile devices.  The goal is to enable mobile devices to tackle complex deep learning oriented-problem heretofore reserved to desktop computers. This project has been initiated by the parallel and distributed processing laboratory at National Taiwan University. TransferCL is released under Mozilla Public Licence 2.0.

### 1. Why TransferCL ?

Recent mobile devices are equipped with multiple sensors, which can give insight into the mobile users' profile.  We believe that such information can be used to customize the mobile experience for a specific user.

The primary goal of TransferCL is to leverage Transfer Learning on mobile devices. Our work is based on the [DeepCL Library](https://github.com/hughperkins/DeepCL). Despite the similarity, TransferCL has been designed to run efficiently on a broad range of mobile devices. As a result, TransferCL implements its own memory management system and own OpenCL kernels in order take into account the specificity of mobile devices' System-on-Chip.

### 2. How does it work?

TransferCL has been implementing in C++ and is able to run on any Android device with an OpenCL compliant GPU (the vast majority of modern Android devices). TransferCL provides several APIs which allow programmers to transparency leverage deep learning on mobile devices.

#### 2.1 Transfer Learning

Modern mobile devices suffer from two major issues that prevent them from training a deep neural network on mobile devices via a classic supervised learning approach. 

* First, the computing capabilities are relatively limited in comparison to the one on servers.
* Then, a single mobile device may not have a sufficient label data in it training data set to train a deep neural network accurately

![file architecture](/image/traditional_ml_setup.png?raw=true)

Transfer learning is a technique that shortcuts a lot of this work by taking a fully-trained model for a set of categories like ImageNet and retrains from the existing weights for new classes.  Using pre-trained features is currently the most straightforward and most commonly used way to perform transfer learning. However, it is by far not the only one.

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
* TransferCL has been pre-build for several commonly used hardware configurations. As a result, for these configurations, the shared-library can be imported directly in the android application. However, we emphasize that once built a shared-library is specific to a CPU ABI (armeabi-v7a, arm64-v8a ...) a GPU architecture (Adreno, Mali ...) and won't work for any other configurations than the one targeted initially.

#### 3.1 Building from source: Native Library installation

##### 3.1.1 Pre-requisites

* OpenCL compliant GPU, along with appropriate OpenCL driver:
    * The ```libOpenCL.so```, corresponding to the mobile device's GPU which is being targeting, need to to be placed in the folder ```extra_libs```.
    * the headers files (*.h) need to be placed in the folder ```include``` 
    
* CrystaX NDK: 
    * [Google NDK](https://developer.android.com/ndk/index.html) provides a set of tools to build native applications on Android.  Our work is based on [CrystaX NDK](https://www.crystax.net/en), which has been developed as a drop-in replacement for Google NDK. For more information, please check their [website](https://www.crystax.net/en).
    * It is still possible to use Google NDK, however, the user will need the import 'Boost C++' by itself.
	
##### 3.1.1.1 Where to find the appropriated OpenCL shared-library

As mentioned previously, the installation of TransferCL requires the compatible libOpenCL.so library and the corresponding OpenCL headers:
* The headers: the slimplest way is to from extract them from Adreno/Mali SDK. For Adreno SDK, they can be found at <Adreno_SDK>/Development/Inc/CL. For Mali SDK, they can be found at <MALI_SDK>/include/CL.
* The ```libOpenCL.so```:  the library is generally already present on the mobile device and can be pull via ```adb pull /system/vendor/lib/libOpenCL.so ./```. (the path may change from one brand to another)


##### 3.1.2 Procedure

* git clone https://github.com/OValery16/TransferCL.git
* add your libOpenCL.so in the folder ```extra_libs```.
* add the OpenCL header in the folder ```include```.

Your repository should look like that:

![file architecture](/image/files2.png?raw=true)

* In the folder 'jni', create a ```\*.cpp``` file and a ```&ast.h``` file, which role is to interface with TranferCL. The android application will call this file's method to interact with the deep learning network.
    * An example can be found in ```sonyOpenCLexample1.cpp```
    * The name of the functions need to be modified in order to respect the naming convention for native function in NDK/JNI application: ```Java_{package_and_classname}_{function_name}(JNI arguments)```
        * For example the ```Java_com_sony_openclexample1_OpenCLActivity_training``` means that this method is mapped to the ```training``` method from the  ```OpenCLActivity``` activity in the ```com.sony.openclexample1``` package.
        * For more information about this naming convention, please check this [website](https://www3.ntu.edu.sg/home/ehchua/programming/java/JavaNativeInterface.html)
* In the 'Android.mk', change the line ```LOCAL_SRC_FILES :=sonyOpenCLexample1.cpp``` to ```LOCAL_SRC_FILES :={your_file_name}.cpp``` (replace 'your_file_name' by the name of the file you just created)
* In the 'Application.mk' change the line ```APP_ABI := armeabi-v7a``` to ```APP_ABI := {the_ABI_you_want_to_target}``` (replace 'the_ABI_you_want_to_target' by the ABI you want to target)
    * A list of all supported ABIs are given on the [NDK website](https://developer.android.com/ndk/guides/abis.html).
    * Make sure that your device supports the chosen ABI (otherwise it won't be able to find TransferCL 's methods). If you are not certain, you can check, which ABIs are supported by your device, via some android applications like ```OpenCL-Z```.
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
    public static native int training(String path); 
    public static native int prediction(String path);
    public static native int prepareFiles(String path);
```
* Build your applications

#### 3.2 Installation from prebuild packages

In process

## 4. How to use it

* In the template file (```sonyOpenCLexample1.cpp```), you can find three methods that have been already defined:
    * ```prepareFiles(String path)```
        * This method builds the training data set
        * Originally the training data set is stored on the microSD card as a set of jpeg images and a manifest file as defined on [DeepCL website in the section 'jpegs'](https://github.com/hughperkins/DeepCL/blob/master/doc/Loaders.md)
            * In future versions of this tutorial, there will be some concrete examples.
        * The images are processed by TransferCL and stored on the mobile device as a unique binary file
        * I also create the folder architecture on your mobile device to store pre-build OpenCL kernel.
            * If these folders are not created, the application will crash 
        * This method has to be the first to run.	
    * ```training(String path)```
        * This method trains the new deep neural network. 
        * This method reuse the previously created files.
        * This method also build the OpenCL kernel the system need to train the deep neural network. 
        * The parameters of the training methods are given in 'sonyOpenCLexample1.cpp'
    * ```prediction(String path)```
        * This method performs the inference task and store the result in a text file

* Currently the most convenient way is to use [DeepCL Library](https://github.com/hughperkins/DeepCL) to train the first deep learning model on mobile.
    * However a conversion tool (TensorFlow model/TransferCL) is in preparation.
* A more detailed tutorial is in preparation.

## 5. Case study


* A case study is defined in ```sonyOpenCLexample1.cpp.example```
	1. We train a network (LeNet5) on the server with MNIST dataset (the training configuration is the standard one).
	2. The final model is stored on the server in a binary file. 
	3. This binary file is copied on the mobile device (for example, on the SD Card) .
	4. TransferCL creates a neural network, and initializes the weights of all layers except the last one with the weights of the pre-trained network. 
	5. The last layer is initialized with a random number generator.
	6. The training starts: TransferCL train the final layer from scratch, while leaving all the others untouched.
		1. TransferCL performs the forward propagation
		2. TransferCL performs the backward propagation and the weight update only on the last layer.
	7. After very few iterations, the prediction error is relatively low.
	8. We test our model prediction accuracy with a test dataset, which our model has never seen. In our case, TransferCL predict all test images label correctly.
* To Conclude this case study, TransferCL trained on only about 12 images per class (a total of 10 classes) in a few seconds and predicted all test images correctly.

## 6. Important remark

* The training images must cover sufficiently the scenarios that you want to predict. If the classifier sees fully new concepts or contexts, it is likely to perform badly. It applies in particular when leveraging transfer learning in a mobile device environment.

    * If the training dataset only contains images from a constraint environment (say, indoor), the classifier won't be able to score images accurately from a different environment (outdoor).
    * If the test images have widely different characteristics (illumination, background, color, size, position, etc), the classifier won't be able to perform very well.
    * If a test image contains entirely new concepts, the classifier won't be able to identify its class.
    
* The choice of the base model to transfer to the mobile device is very important. The two classification tasks (the one on the server, and the one on the mobile device) should be related. For example, in our case study, the base network has been trained to recognize handwritten digits, and this knowledge is transferred to TransferCL in order to train a new network to classify handwritten characters on mobile devices.

## 7. How to see the output 

* In order to see the ouput of TranferCL, you can use the [logcat command-line tool](https://developer.android.com/studio/command-line/logcat.html):
'''
>adb logcat ActivityManager:I TransferCL:D *:S
'''
* For example, the output of ```prepareFiles(String path)``` should look like that.

		```
		I/TransferCL(11481): -------Files Preparation
		I/TransferCL(11481): -----------Generation of the memory-map files (binary files)
		I/TransferCL(11481): -------------- 128 images in the training set with 1 dimension and a image size of 28 X 28
		I/TransferCL(11481): -------------- Training set loading
		I/TransferCL(11481): -------------- training data file generation: /data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw
		I/TransferCL(11481): -------------- label file generation: /data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw
		I/TransferCL(11481): -------------- normalization file file generation: /data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt
		I/TransferCL(11481): -------File generation: completed
		I/TransferCL(11481): easyCL oject destroyed
		```

* For example, the output of ```training(String path)``` should look like that.

```
		I/TransferCL(11481): ################################################
		I/TransferCL(11481): ###################Training#####################
		I/TransferCL(11481): ################################################
		I/TransferCL(11481): ------- Loading configuration: training set 128 images with 1 channel and an image size of 28 X 28
		I/TransferCL(11481): ------- Network Generation: 1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
		I/TransferCL(11481): -----------Define Weights Initialization Method
		I/TransferCL(11481): -------------- Chosen Initializer: original
		I/TransferCL(11481): -----------Network Layer Generation
		I/TransferCL(11481): -----------Selecting Training method (sgd)
		I/TransferCL(11481): -----------Loading the weights (the othe weights are radomly initialized with the initializer defined previously)
		I/TransferCL(11481): -----------Set up Trainer
		I/TransferCL(11481):
		I/TransferCL(11481):
		I/TransferCL(11481): ################################################
		I/TransferCL(11481): ################Start learning##################
		I/TransferCL(11481): ################################################
		I/TransferCL(11481):
		I/TransferCL(11481):
		I/TransferCL(11481): loss=339.530243 numRight=7
		I/TransferCL(11481): loss=103.915398 numRight=100
		I/TransferCL(11481): loss=736.500610 numRight=64
		I/TransferCL(11481): loss=1662.703247 numRight=64
		I/TransferCL(11481): loss=68.743599 numRight=115
		I/TransferCL(11481): loss=19.016699 numRight=125
		I/TransferCL(11481): loss=11.302835 numRight=125
		I/TransferCL(11481): loss=6.223648 numRight=125
		I/TransferCL(11481): loss=4.286274 numRight=126
		I/TransferCL(11481): loss=2.810870 numRight=127
		I/TransferCL(11481): loss=1.905059 numRight=128
		I/TransferCL(11481): loss=1.448894 numRight=128
		I/TransferCL(11481): loss=1.189890 numRight=128
		I/TransferCL(11481): loss=1.017845 numRight=128
		I/TransferCL(11481): loss=0.892899 numRight=128
		I/TransferCL(11481): loss=0.797294 numRight=128
		I/TransferCL(11481): loss=0.721556 numRight=128
		I/TransferCL(11481): loss=0.659997 numRight=128
		I/TransferCL(11481): loss=0.608945 numRight=128
		I/TransferCL(11481): loss=0.565904 numRight=128
		I/TransferCL(11481): loss=0.529109 numRight=128
		I/TransferCL(11481): loss=0.497274 numRight=128
		I/TransferCL(11481): loss=0.469449 numRight=128
		I/TransferCL(11481): loss=0.444901 numRight=128
		I/TransferCL(11481): loss=0.423074 numRight=128
		I/TransferCL(11481): loss=0.403524 numRight=128
		I/TransferCL(11481): loss=0.385904 numRight=128
		I/TransferCL(11481): loss=0.369931 numRight=128
		I/TransferCL(11481): loss=0.355375 numRight=128
		I/TransferCL(11481): loss=0.342049 numRight=128
		I/TransferCL(11481): loss=0.329797 numRight=128
		I/TransferCL(11481): loss=0.318488 numRight=128
		I/TransferCL(11481): loss=0.308013 numRight=128
		I/TransferCL(11481): loss=0.298278 numRight=128
		I/TransferCL(11481): loss=0.289205 numRight=128
		I/TransferCL(11481): loss=0.280725 numRight=128
		I/TransferCL(11481): loss=0.272778 numRight=128
		I/TransferCL(11481): loss=0.265314 numRight=128
		I/TransferCL(11481): loss=0.258288 numRight=128
		I/TransferCL(11481): loss=0.251661 numRight=128
		I/TransferCL(11481): loss=0.245398 numRight=128
		I/TransferCL(11481): loss=0.239468 numRight=128
		I/TransferCL(11481): loss=0.233845 numRight=128
		I/TransferCL(11481): loss=0.228503 numRight=128
		I/TransferCL(11481): loss=0.223422 numRight=128
		I/TransferCL(11481): loss=0.218584 numRight=128
		I/TransferCL(11481): loss=0.213968 numRight=128
		I/TransferCL(11481): loss=0.209559 numRight=128
		I/TransferCL(11481): loss=0.205345 numRight=128
		I/TransferCL(11481): loss=0.201312 numRight=128
		I/TransferCL(11481): loss=0.197446 numRight=128
		I/TransferCL(11481): loss=0.193739 numRight=128
		I/TransferCL(11481): loss=0.190181 numRight=128
		I/TransferCL(11481): loss=0.186761 numRight=128
		I/TransferCL(11481): loss=0.183472 numRight=128
		I/TransferCL(11481): loss=0.180306 numRight=128
		I/TransferCL(11481): loss=0.177256 numRight=128
		I/TransferCL(11481): loss=0.174317 numRight=128
		I/TransferCL(11481): loss=0.171480 numRight=128
		I/TransferCL(11481): loss=0.168741 numRight=128
		I/TransferCL(11481): loss=0.166096 numRight=128
		I/TransferCL(11481): loss=0.163539 numRight=128
		I/TransferCL(11481): loss=0.161066 numRight=128
		I/TransferCL(11481): loss=0.158672 numRight=128
		I/TransferCL(11481): loss=0.156354 numRight=128
		I/TransferCL(11481): loss=0.154107 numRight=128
		I/TransferCL(11481): loss=0.151930 numRight=128
		I/TransferCL(11481): loss=0.149818 numRight=128
		I/TransferCL(11481): loss=0.147768 numRight=128
		I/TransferCL(11481): loss=0.145778 numRight=128
		I/TransferCL(11481): loss=0.143844 numRight=128
		I/TransferCL(11481): loss=0.141966 numRight=128
		I/TransferCL(11481): loss=0.140139 numRight=128
		I/TransferCL(11481): loss=0.138362 numRight=128
		I/TransferCL(11481): loss=0.136634 numRight=128
		I/TransferCL(11481): loss=0.134952 numRight=128
		I/TransferCL(11481): loss=0.133313 numRight=128
		I/TransferCL(11481): loss=0.131717 numRight=128
		I/TransferCL(11481): loss=0.130161 numRight=128
		I/TransferCL(11481): loss=0.128645 numRight=128
		I/TransferCL(11481): loss=0.127166 numRight=128
		I/TransferCL(11481): loss=0.125724 numRight=128
		I/TransferCL(11481): loss=0.124316 numRight=128
		I/TransferCL(11481): loss=0.122942 numRight=128
		I/TransferCL(11481): loss=0.121600 numRight=128
		I/TransferCL(11481): loss=0.120290 numRight=128
		I/TransferCL(11481): loss=0.119010 numRight=128
		I/TransferCL(11481): loss=0.117759 numRight=128
		I/TransferCL(11481): loss=0.116535 numRight=128
		I/TransferCL(11481): loss=0.115339 numRight=128
		I/TransferCL(11481): loss=0.114169 numRight=128
		I/TransferCL(11481): loss=0.113024 numRight=128
		I/TransferCL(11481): loss=0.111905 numRight=128
		I/TransferCL(11481): loss=0.110808 numRight=128
		I/TransferCL(11481): loss=0.109735 numRight=128
		I/TransferCL(11481): loss=0.108684 numRight=128
		I/TransferCL(11481): loss=0.107654 numRight=128
		I/TransferCL(11481): loss=0.106646 numRight=128
		I/TransferCL(11481): loss=0.105658 numRight=128
		I/TransferCL(11481): loss=0.104689 numRight=128
		I/TransferCL(11481): gettimeofday 3723.000000
		I/TransferCL(11481):  ms
		I/TransferCL(11481): -----------End of ther training: Delete object
		I/TransferCL(11481): -----------Delete weightsInitializer
		I/TransferCL(11481): -----------Delete trainer
		I/TransferCL(11481): -----------Delete netLearner
		I/TransferCL(11481): -----------Delete net
		I/TransferCL(11481): -----------Delete trainLoader
		I/TransferCL(11481): finish
		I/TransferCL(11481): finish1
		I/TransferCL(11481): All code took 3797.000000
		I/TransferCL(11481):  ms
		I/TransferCL(11481): easyCL oject destroyed
		I/TransferCL(11481): 3)time 3.834609
		I/TransferCL(11481):
		I/TransferCL(11481): 3834.000000
		I/TransferCL(11481):  ms
```

* For example, the output of ```prediction(String path)``` should look like that.

```
		I/TransferCL(11481): ################################################
		I/TransferCL(11481): ###################Prediction###################
		I/TransferCL(11481): ################################################
		I/TransferCL(11481): ------- Network Generation
		I/TransferCL(11481): -----------Network Layers Creation 1s8c5z-relu-mp2-1s16c5z-relu-mp3-152n-tanh-10n
		I/TransferCL(11481): -----------Loading the weights
		I/TransferCL(11481): -----------Start prediction
		I/TransferCL(11481): --------- Prediction: done (prediction in /data/data/com.sony.openclexample1/preloadingData/pred2.txt)
		I/TransferCL(11481): --------- End of ther prediction: Delete objects
		I/TransferCL(11481): easyCL oject destroyed
```


## 8. To get in contact

Just create issues, in GitHub, in the top right of this page. Don't worry about whether you think your issue sounds silly or anything. The more feedback, the better!

## 9. Contribute

If you are interestered in this project, feel free to contact me.









