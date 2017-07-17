## TransferCL

TransferCL is an open source deep learning framework which has been designed to run on mobile devices.  The goal is to enable mobile devices to tackle complex deep learning oriented*problem heretofore reserved to desktop computers. This project has been initiated by the parallel and distributed processing laboratory at National Taiwan University. TransferCL is released under Mozilla Public Licence 2.0.

### Why TransferCL ?

Recent mobile devices are equipped with multiple sensors, which can give insight into the mobile users' profile.  We believe that such information can be used to customize the mobile experience for a specific user.

The primary goal of TransferCL is to leverage Transfer Learning on mobile devices. Our work is based on the [DeepCL Library](https://github.com/hughperkins/DeepCL). Despite the similarity, TransferCL has been designed to run efficiently on a broad range of mobile devices. As a result, TransferCL implements its own memory management system and own OpenCL kernels in order take into account the specificity of mobile devices' System-on-Chip.

### How does it work?

TransferCL has been implementing in C++ and is able to run on any Android device with an OpenCL compliant GPU (the vast majority of modern Android devices). TransferCL provides several APIs which allow programmers to transparency leverage deep learning on mobile devices.

#### Transfer Learning

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

### Installation

#### Native Library installation

#### Pre-requisites

* OpenCL compliant GPU, along with appropriate OpenCL driver:
    * The 'libOpenCL.so', corresponding to the mobile device's GPU which is being targeting, need to to be placed in the folder 'extra_libs'.
    * the headers files (*.h) need to be placed in the folder 'include' 
    
* CrystaX NDK: 
    * [Google NDK](https://developer.android.com/ndk/index.html) provides a set of tools to build native applications on Android.  Our work is based on [CrystaX NDK](https://www.crystax.net/en), which has been developed as a drop-in replacement for Google NDK. For more information, please check their [website](https://www.crystax.net/en).
    * It is still possible to use Google NDK, however, the user will need the import 'Boost C++' by itself.

#### Procedure

* git clone https://github.com/OValery16/TransferCL.git
* add your libOpenCL.so in the folder 'extra_libs'.
* add the OpenCL header in the folder 'include'.

Your repository should look like that:

![file architecture](/image/files2.png?raw=true)

* In the folder 'jni', create a '\*.cpp' file and a '&ast.h' file, which role is to interface with TranferCL. The android application will call this file's method to interact with the deep learning network.
    * An example can be found in 'sonyOpenCLexample1.cpp'
    * The name of the functions need to be modified in order to respect the naming convention for native function in NDK/JNI application: 'Java_{package_and_classname}_{function_name}(JNI arguments)'
        * For example the 'Java_com_sony_openclexample1_OpenCLActivity_training' means that this method is mapped to the 'training' method from the  'OpenCLActivity' activity in the 'com.sony.openclexample1' package.
        * For more information about this naming convention, please check this [website](https://www3.ntu.edu.sg/home/ehchua/programming/java/JavaNativeInterface.html)
* In the 'Android.mk', change the line 'LOCAL_SRC_FILES :=sonyOpenCLexample1.cpp' to 'LOCAL_SRC_FILES :={your_file_name}.cpp' (replace 'your_file_name' by the name of the file you just created)
* In the 'Application.mk' change the line 'APP_ABI := armeabi-v7a' to 'APP_ABI := {the_ABI_you_want_to_target}' (replace 'the_ABI_you_want_to_target' by the ABI you want to target)
    * A list of all supported ABIs are given on the [NDK website](https://developer.android.com/ndk/guides/abis.html).
    * Make sure that your device supports the chosen ABI (otherwise it won't be able to find TransferCL 's methods). If you are not certain, you can check, which ABIs are supported by your device, via some android applications like 'OpenCL-Z'.
* Run CrystaX  NDK to build your shared library with the command 'ndk-build' (crystax-ndk-X\ndk-build where X is CrystaX NDK version)
```
>ndk-build
```
* CrystaX NDK will output several shared library files (they are specific to your mobile device ABIs)

#### Android application installation

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

## How to use it

* In the template file ('sonyOpenCLexample1.cpp'), you can find three methods that have been already defined:
    * *prepareFiles(String path)*
        * This method builds the training data set
        * Originally the training data set is stored on the microSD card as a set of jpeg images and a manifest file as defined on [DeepCL website in section 'jpegs'](https://github.com/hughperkins/DeepCL/blob/master/doc/Loaders.md)
            * In future versions of this tutorial, there will be some concrete examples.
        * The images are processed by TransferCL and stored on the mobile device as a unique binary file
        * I also create the folder architecture on your mobile device to store pre-build OpenCL kernel.
            * If these folders are not created, the application will crash 
        * This method has to be the first to run.	
    * *training(String path)*
        * This method trains the new deep neural network. 
        * This method reuse the previously created files.
        * This method also build the OpenCL kernel the system need to train the deep neural network. 
        * The parameters of the training methods are given in 'sonyOpenCLexample1.cpp'
    * *prediction(String path)*
        * This method performs the inference task and store the result in a text file

* Currently the most convenient way is to use [DeepCL Library](https://github.com/hughperkins/DeepCL) to train the first deep learning model on mobile.
    * However a conversion tool (TensorFlow model/TransferCL) is in preparation.
* A more detailed tutorial is in preparation.

## How to see the output 

* In order to see the ouput of TranferCL, you can use the [logcat command-line tool](https://developer.android.com/studio/command-line/logcat.html):
'''
>adb logcat ActivityManager:I TransferCL:D *:S
'''
* For example, the output of *prepareFiles(String path)* should look like that.

		'''
		I/TransferCL(11481): -------Files Preparation
		
		I/TransferCL(11481): -----------Generation of the memory-map files (binary files)
		
		I/TransferCL(11481): -------------- 128 images in the training set with 1 dimension and a image size of 28 X 28
		
		I/TransferCL(11481): -------------- Training set loading
		
		I/TransferCL(11481): -------------- training data file generation: 
		
		/data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileData2.raw
		
		I/TransferCL(11481): -------------- label file generation: /data/data/com.sony.openclexample1/directoryTest/mem2Character2ManifestMapFileLabel2.raw
		
		I/TransferCL(11481): -------------- normalization file file generation: /data/data/com.sony.openclexample1/directoryTest/normalizationTranfer.txt
		
		I/TransferCL(11481): -------File generation: completed
		
		I/TransferCL(11481): easyCL oject destroyed
		
		'''
		
* For example, the output of *training(String path)* should look like that.

'''
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
		I/TransferCL(11481): loss=298.027008 numRight=1
		I/TransferCL(11481): loss=86.451286 numRight=67
		I/TransferCL(11481): loss=893.552063 numRight=64
		I/TransferCL(11481): loss=815.358154 numRight=64
		I/TransferCL(11481): loss=640.962585 numRight=64
		I/TransferCL(11481): loss=868.909058 numRight=64
		I/TransferCL(11481): loss=390.025848 numRight=64
		I/TransferCL(11481): loss=888.694336 numRight=64
		I/TransferCL(11481): loss=209.329407 numRight=76
		I/TransferCL(11481): loss=473.083618 numRight=64
		I/TransferCL(11481): loss=442.481567 numRight=67
		I/TransferCL(11481): loss=493.395844 numRight=65
		I/TransferCL(11481): loss=250.781067 numRight=80
		I/TransferCL(11481): loss=196.740875 numRight=95
		I/TransferCL(11481): loss=62.101933 numRight=116
		I/TransferCL(11481): loss=39.748695 numRight=120
		I/TransferCL(11481): loss=36.382439 numRight=120
		I/TransferCL(11481): loss=33.396393 numRight=120
		I/TransferCL(11481): loss=30.748188 numRight=122
		I/TransferCL(11481): loss=28.397659 numRight=122
		I/TransferCL(11481): loss=26.303474 numRight=123
		I/TransferCL(11481): loss=24.424046 numRight=123
		I/TransferCL(11481): loss=22.720396 numRight=124
		I/TransferCL(11481): loss=21.158806 numRight=124
		I/TransferCL(11481): loss=19.711933 numRight=124
		I/TransferCL(11481): loss=18.358883 numRight=124
		I/TransferCL(11481): loss=17.084633 numRight=124
		I/TransferCL(11481): loss=15.879351 numRight=125
		I/TransferCL(11481): loss=14.737835 numRight=125
		I/TransferCL(11481): loss=13.658861 numRight=125
		I/TransferCL(11481): loss=12.644492 numRight=125
		I/TransferCL(11481): loss=11.698782 numRight=125
		I/TransferCL(11481): loss=10.826109 numRight=125
		I/TransferCL(11481): loss=10.029263 numRight=125
		I/TransferCL(11481): loss=9.308178 numRight=125
		I/TransferCL(11481): loss=8.659803 numRight=125
		I/TransferCL(11481): loss=8.078922 numRight=125
		I/TransferCL(11481): loss=7.559265 numRight=126
		I/TransferCL(11481): loss=7.094359 numRight=126
		I/TransferCL(11481): loss=6.678047 numRight=126
		I/TransferCL(11481): loss=6.304677 numRight=126
		I/TransferCL(11481): loss=5.969164 numRight=126
		I/TransferCL(11481): loss=5.666996 numRight=126
		I/TransferCL(11481): loss=5.394205 numRight=126
		I/TransferCL(11481): loss=5.147293 numRight=126
		I/TransferCL(11481): loss=4.923184 numRight=126
		I/TransferCL(11481): loss=4.719198 numRight=126
		I/TransferCL(11481): loss=4.532969 numRight=126
		I/TransferCL(11481): loss=4.362426 numRight=126
		I/TransferCL(11481): loss=4.205747 numRight=126
		I/TransferCL(11481): loss=4.061347 numRight=126
		I/TransferCL(11481): loss=3.927827 numRight=126
		I/TransferCL(11481): loss=3.803984 numRight=126
		I/TransferCL(11481): loss=3.688763 numRight=126
		I/TransferCL(11481): loss=3.581251 numRight=126
		I/TransferCL(11481): loss=3.480651 numRight=126
		I/TransferCL(11481): loss=3.386279 numRight=127
		I/TransferCL(11481): loss=3.297540 numRight=127
		I/TransferCL(11481): loss=3.213910 numRight=127
		I/TransferCL(11481): loss=3.134939 numRight=127
		I/TransferCL(11481): loss=3.060223 numRight=127
		I/TransferCL(11481): loss=2.989419 numRight=127
		I/TransferCL(11481): loss=2.922213 numRight=127
		I/TransferCL(11481): loss=2.858338 numRight=127
		I/TransferCL(11481): loss=2.797545 numRight=127
		I/TransferCL(11481): loss=2.739614 numRight=127
		I/TransferCL(11481): loss=2.684350 numRight=127
		I/TransferCL(11481): loss=2.631572 numRight=127
		I/TransferCL(11481): loss=2.581122 numRight=127
		I/TransferCL(11481): loss=2.532857 numRight=127
		I/TransferCL(11481): loss=2.486634 numRight=127
		I/TransferCL(11481): loss=2.442336 numRight=128
		I/TransferCL(11481): loss=2.399849 numRight=128
		I/TransferCL(11481): loss=2.359071 numRight=128
		I/TransferCL(11481): loss=2.319901 numRight=128
		I/TransferCL(11481): loss=2.282254 numRight=128
		I/TransferCL(11481): loss=2.246046 numRight=128
		I/TransferCL(11481): loss=2.211201 numRight=128
		I/TransferCL(11481): loss=2.177646 numRight=128
		I/TransferCL(11481): loss=2.145315 numRight=128
		I/TransferCL(11481): loss=2.114148 numRight=128
		I/TransferCL(11481): loss=2.084084 numRight=128
		I/TransferCL(11481): loss=2.055071 numRight=128
		I/TransferCL(11481): loss=2.027053 numRight=128
		I/TransferCL(11481): loss=1.999988 numRight=128
		I/TransferCL(11481): loss=1.973827 numRight=128
		I/TransferCL(11481): loss=1.948528 numRight=128
		I/TransferCL(11481): loss=1.924053 numRight=128
		I/TransferCL(11481): loss=1.900362 numRight=128
		I/TransferCL(11481): loss=1.877420 numRight=128
		I/TransferCL(11481): loss=1.855195 numRight=128
		I/TransferCL(11481): loss=1.833654 numRight=128
		I/TransferCL(11481): loss=1.812766 numRight=128
		I/TransferCL(11481): loss=1.792504 numRight=128
		I/TransferCL(11481): loss=1.772841 numRight=128
		I/TransferCL(11481): loss=1.753754 numRight=128
		I/TransferCL(11481): loss=1.735215 numRight=128
		I/TransferCL(11481): loss=1.717201 numRight=128
		I/TransferCL(11481): loss=1.699694 numRight=128
		I/TransferCL(11481): loss=1.682672 numRight=128
		I/TransferCL(11481): gettimeofday 4119.000000
		I/TransferCL(11481):  ms
		I/TransferCL(11481): -----------End of ther training: Delete object
		I/TransferCL(11481): -----------Delete weightsInitializer
		I/TransferCL(11481): -----------Delete trainer
		I/TransferCL(11481): -----------Delete netLearner
		I/TransferCL(11481): -----------Delete net
		I/TransferCL(11481): -----------Delete trainLoader
		I/TransferCL(11481): finish
		I/TransferCL(11481): finish1
		I/TransferCL(11481): All code took 4158.000000
		I/TransferCL(11481):  ms
		I/TransferCL(11481): easyCL oject destroyed
		I/TransferCL(11481): 3)time 4.181477
		I/TransferCL(11481):
		I/TransferCL(11481): 4181.000000
		I/TransferCL(11481):  ms
'''
* For example, the output of *prediction(String path)* should look like that.
'''
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
'''



## To get in contact

Just create issues, in GitHub, in the top right of this page. Don't worry about whether you think your issue sounds silly or anything. The more feedback, the better!

## Contribute

If you are interestered in this project, feel free to contact me.










