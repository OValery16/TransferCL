LOCAL_PATH		:= $(call my-dir)
LOCAL_PATH_EXT	:= $(call my-dir)/../extra_libs/
MY_PATH := $(LOCAL_PATH)
include $(call all-subdir-makefiles)

include $(CLEAR_VARS)

LOCAL_PATH := $(MY_PATH)

include $(CLEAR_VARS)

LOCAL_ARM_MODE  := arm

LOCAL_MODULE    := transferCL

LOCAL_CFLAGS 	+= -DANDROID_CL
LOCAL_CFLAGS    += -O3 -ffast-math

LOCAL_C_INCLUDES := $(LOCAL_PATH)/../include 

LOCAL_EXPORT_LDLIBS := -latomic
LOCAL_SRC_FILES :=trainEngine/train.cpp predictEngine/predict.cpp 
LOCAL_SRC_FILES += $(notdir $(LOCAL_PATH)/TransferCL)/src/netdef/NetdefToNet.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loaders/GenericLoaderv2.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loaders/ManifestLoaderv1.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loaders/Loader.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loaders/GenericLoaderv1Wrapper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loaders/GenericLoader.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingBackwardCpu.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingForwardCpu.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingBackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingBackward.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingForwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/pooling/PoolingForward.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/util/JpegHelper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/util/FileHelper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/util/stringhelper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/util/RandomSingleton.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/net/Trainable.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/net/NeuralNetMould.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/net/NeuralNet.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/TransferCL.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/NetAction.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/NetAction2.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/NetLearnerOnDemandv2.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/OnDemandBatcher.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/BatchData.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/EpochMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/Batcher.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/OnDemandBatcherv2.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/BatchLearnerOnDemand.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/Batcher2.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/BatchProcess.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/NetLearner.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/batch/NetLearnerOnDemand.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/input/InputLayerMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/input/InputLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/normalize/NormalizationLayerMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/normalize/NormalizationLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/LossLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/CrossEntropyLoss.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/SoftMaxLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/SoftMaxLayerPredict.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/SquareLossLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/loss/IAcceptsLabels.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/layer/Layer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/layer/LayerMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/SGDMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/TrainingContext.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/TrainerState.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/Trainer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/SGDStateMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/TrainerStateMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/SGD.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/SGDState.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/trainers/TrainerMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/activate/ActivationLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/activate/ActivationMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/weights/WeightsPersister.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/weights/WeightsInitializer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/weights/OriginalInitializer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/weights/UniformInitializer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/CppRuntimeBoundary.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/BackpropWeightsNaive.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/ConvolutionalLayer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/BackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/ConvolutionalMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/LayerDimensions.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/conv/Forward1.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/fc/FullyConnectedMaker.cpp $(notdir $(LOCAL_PATH)/TransferCL)/src/fc/FullyConnectedLayer.cpp
LOCAL_SRC_FILES +=$(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/luac.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ltablib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lbaselib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/loslib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/loadlib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lstrlib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lapi.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lmathlib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ldump.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lzio.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ldblib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lgc.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lmem.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/liolib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lopcodes.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ltm.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lparser.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ltable.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/linit.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lauxlib.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lua.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ldebug.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lundump.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/llex.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/print.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lvm.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lstate.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/ldo.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lfunc.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lstring.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lcode.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/thirdparty/lua-5.1.5/src/lobject.c $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/util/StatefulTimer.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/util/easycl_stringhelper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/CLKernel.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/deviceinfo_helper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/DevicesInfo.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/gpuinfo.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/DeviceInfo.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/EasyCL.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/templates/LuaTemplater.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/templates/TemplatedKernel.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/platforminfo_helper.cpp $(notdir $(LOCAL_PATH)/TransferCL)/EasyCL/CLWrapper.cpp
LOCAL_SRC_FILES += $(notdir $(LOCAL_PATH)/)kernelManager/ConfigManager.cpp 

#$(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationFunction.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForwardGpuNaive.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackward.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationBackwardCpu.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/activate/ActivationForwardCpu.cpp
#$(notdir $(LOCAL_PATH)/DeepCL)/src/forcebackprop/ForceBackpropLayer.cpp $(notdir $(LOCAL_PATH)/DeepCL)/src/forcebackprop/ForceBackpropLayerMaker.cpp  
LOCAL_LDLIBS 	:= -llog -ljnigraphics 
LOCAL_SHARED_LIBRARIES := libjpegturboSIMD 
#LOCAL_SHARED_LIBRARIES := libjpeg_shared
 

LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp

LOCAL_LDLIBS := -ljnigraphics -llog -landroid
LOCAL_C_INCLUDES += $(LOCAL_PATH) \
                    $(LOCAL_PATH)/libjpegturbo \
                    $(LOCAL_PATH)/libjpegturbo/android             
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

LOCAL_STATIC_LIBRARIES :=boost_iostreams_static
#LOCAL_STATIC_LIBRARIES += libjpeg-turbo
#LOCAL_STATIC_LIBRARIES += libjpeg_static_no_neon
# LOCAL_SHARED_LIBRARIES := libDeepCL libEasyCL

LOCAL_LDLIBS 	:= -llog -ljnigraphics 
LDFLAGS += -pthread
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_ARM_MODE  := arm

LOCAL_MODULE    := transferCLNative

LOCAL_CFLAGS 	+= -DANDROID_CL
LOCAL_CFLAGS    += -O3 -ffast-math
LOCAL_LDLIBS := -ljnigraphics -llog -landroid
LOCAL_CFLAGS += -fopenmp
LOCAL_LDFLAGS += -fopenmp
LOCAL_SRC_FILES :=transferCLinterface.cpp
#sonyOpenCLexample1.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)/../include 
LOCAL_STATIC_LIBRARIES :=transferCL
LOCAL_LDLIBS 	+= $(LOCAL_PATH_EXT)libOpenCL.so  

include $(BUILD_SHARED_LIBRARY)

$(call import-module,boost/1.57.0)
$(call import-module,libjpeg/9a)
