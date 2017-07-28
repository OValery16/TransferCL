## Android Application

### 1. Description

* We also provide a template application using TranferCL. This application defines 2 Java source package:
    * ```com.transferCL```, which is a java wrapper for the native methods defined in TranferCL (```TransferCLlib.java```).
    * ```com.example.myapplication```, which is an android activity (```MainActivity.java```). It calls the methods declared in ```TransferCLlib.java```.

* We emphasize the fact that TranferCL's methods are declared directly in the android application, but the implementation of these methods are done at the native level. The TransferCL library needs to be built for a specific CPU architecture the user is targeting, such as armeabi-v7a,  and a specific brand of GPU, such as Adreno (Qualcomm).
    * In this [file](../README.md), the developer can find a guide relative the building process.
    * Once the shared-library is built, the developer needs to put the ```.so``` file in the ```jniLibs``` folder, as shown in the picture.
        * Important: 
            1. In the given an example (on the picture), TranferCL has been compiled for ```armeabi``` architecture. The building process will output a ```armeabi``` folder with two files ```libcrystax.so``` and ```libtransferCLNative.so```. This folder needs to be copied in the ```jniLibs```.
            2. Don't change the folder name and the file names.
            3. A list of all supported ABIs is given on the [NDK website](https://developer.android.com/ndk/guides/abis.html). 
    
    
![file architecture](/image/jniLibs.PNG?raw=true)

### 2. Hardware requirements

* In this template application, we target mobile phone which supports ```armeabi``` and/or ```armeabi-v7a``` and that have an Adreno GPU. For any other configuration, you need to build the application from scratch.
    * If you meet any problem during the building process, feel free to create issues (in GitHub) in the top right of this page. Don't worry about whether you think your issue sounds unimportant or trivial. The more feedback we can get, the better!
    
### 3. The application

* This application is relatively simple, there are only three buttons (```prepare files```, ```training``` and ```prediction```) and one TextView that mirrors the adb logcat output.
* Files preparations (```prepare files```)
        1. We create the working directory ```directoryTest``` (perform at the native level by TransferCL)
        2. The training files (the training file and their labels are respectively stored in one binary file) are generated.
        3. TransferCL analyse the dataset, stores its mean/stdDev and store them in one file

![file architecture](/image/filePreparation.png?raw=true)    
        
* Training on the mobile device (```training```)
        1. TransferCL creates a neural network, and initializes the weights of all layers except the last one with the weights of the pre-trained network. 
        2. The last layer is initialized with a random number generator.
        3. The training starts: TransferCL train the final layer from scratch, while leaving all the others untouched.
            1. TransferCL performs the forward propagation.
            2. TransferCL performs the backward propagation and the weight update only on the last layer.
        4. After very few iterations, the prediction error drops significantly. All images' label are predicted correctly after only 11 iterations. (```loss=1.905059 numRight=128```)
        
![file architecture](/image/training1.png?raw=true)        
    
![file architecture](/image/training2.png?raw=true)        

![file architecture](/image/training3.png?raw=true)        

![file architecture](/image/training4.png?raw=true)        
        
* Test on the mobile device (```prediction```)
        1. We tested our model prediction accuracy with a test dataset, which our model has never seen. In our expleriment, TransferCL predicted all test images label correctly.
		
![file architecture](/image/prediction.png?raw=true)        