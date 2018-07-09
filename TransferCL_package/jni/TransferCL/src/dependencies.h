
#ifndef DEPENDENCIES_H
#define DEPENDENCIES_H

#include <time.h>
#include <math.h>


#include <android/log.h>
#define app_name "TransferCL"
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, app_name, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, app_name, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, app_name, __VA_ARGS__))

//these macros define the behavior of the application
// NEVER MODIFY THEIR VALUE IF YOU DON T HAVE A FULL UNDERSTANDING OF WHAT YOU ARE DOING
#define NO_POSTPROCESSING 1
#define MEMORY_MAP_FILE_LOADING 1
#define SAVENETWORK 0
#define TRANSFER 1
#define DISPLAY_LOSS 1
#define DISPLAY_WEIGHT 0
#define TRANSFERCL_VERBOSE 0


#define MNIST_TEST 1
#define VGG16_TEST 0
#define TEST_MEM 0

#endif
