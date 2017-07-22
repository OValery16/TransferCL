## IMPORTANT

* The file ```libOpenCL.so.example``` corresponding to the OpenCL driver of your mobile device.
* "We emphasize that ```libOpenCL.so``` is specific to the architecture (Adreno, Mali ...) and won't work for any other configurations than the one targeted initially.
* The file ```libOpenCL.so.example``` has to be replaced the one corresponding to the mobile device's GPU which is being targeted. 
* The easiest way to get it is to download it from the device itself:
    * The library is generally already present on the mobile device and can be pulled via ```adb pull /system/vendor/lib/libOpenCL.so .```. (the path may change from one brand to another)
