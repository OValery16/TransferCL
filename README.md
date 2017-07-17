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

## To get in contact

Just create issues, in GitHub, in the top right of this page. Don't worry about whether you think your issue sounds silly or anything. The more feedback, the better!

## Contribute

If you are interestered in this project, feel free to contact me.










