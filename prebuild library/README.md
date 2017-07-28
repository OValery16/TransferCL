Installation from prebuild packages
========


### 1. Hardware requirements

* Our prebuild packages target mobile phone which supports ```armeabi``` and/or ```armeabi-v7a``` and that have an Adreno GPU. For any other configuration, you need to build the application from scratch.
    * If you meet any problem during the building process, feel free to create issues (in GitHub) in the top right of this page. Don't worry about whether you think your issue sounds unimportant or trivial. The more feedback we can get, the better!

### 2. Installation	
	
* We emphasize the fact that TranferCL's methods are declared directly in the android application, but the implementation of these methods is done at the native level. The TransferCL library needs to be built for a specific CPU architecture the user is targeting, such as armeabi-v7a,  and a specific brand of GPU, such as Adreno (Qualcomm).
    * In this [file](../README.md), the developer can find a guide relative the building process.
    * Once the shared-library is built, the developer needs to put the ```.so``` file in the ```jniLibs``` folder, as shown in the picture (```case study/android application/MyApplication/```).
        * Important: 
            1. In the given an example (on the picture), TranferCL has been compiled for ```armeabi``` architecture. The building process will output a ```armeabi``` folder with two files ```libcrystax.so``` and ```libtransferCLNative.so```. This folder needs to be copied in the ```jniLibs```.
            2. Don't change the folder name and the file names.
            3. A list of all supported ABIs is given on the [NDK website](https://developer.android.com/ndk/guides/abis.html). 
    
    
![file architecture](/image/jniLibs.PNG?raw=true)	