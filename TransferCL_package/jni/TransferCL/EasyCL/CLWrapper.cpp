// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "EasyCL.h"

#include "CLWrapper.h"
#include "util/easycl_stringhelper.h"

CLWrapper::CLWrapper(int N, EasyCL *cl) : N(N), onHost(true), cl(cl) {
    error = CL_SUCCESS;
    onDevice = false;
    deviceDirty = false;
}
CLWrapper::CLWrapper(const CLWrapper &source) :
     N(0), onHost(true)
        { // copy constructor
    throw std::runtime_error("can't assign these...");
}
CLWrapper &CLWrapper::operator=(const CLWrapper &two) { // assignment operator
   if(this == &two) { // self-assignment
      return *this;
   }
   throw std::runtime_error("can't assign these...");
}
CLWrapper::~CLWrapper() {
    if(onDevice) {
//            std::cout << "releasing device array of " << N << " elements" << std::endl;
        clReleaseMemObject(devicearray);
    }
}
EasyCL *CLWrapper::getCl() {
    return cl;
}
void CLWrapper::deleteFromDevice(){
    if(!onDevice) {
        throw std::runtime_error("deletefromdevice(): not on device");
    }
//        std::cout << "deleted device array of " << N << " elements" << std::endl;
    clReleaseMemObject(devicearray);        
    onDevice = false;
    deviceDirty = false;
}
cl_mem *CLWrapper::getDeviceArray() {
    if(!onDevice) {
        if(!onHost) {
            throw std::runtime_error("getDeviceArray(): not on device, and not on host");
        }
//            std::cout << "copy array to device of " << N << " elements" << std::endl;
        copyToDevice();
    }
    return &devicearray;
}
void CLWrapper::createOnDevice() {
    if(onDevice) {
        throw std::runtime_error("createOnDevice(): already on device");
    }
//        std::cout << "creating buffer on device of " << N << " elements" << std::endl;
    devicearray = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE, getElementSize() * N, 0, &error);
    cl->checkError(error);
    onDevice = true;
    deviceDirty = false;
//        std::cout << "... created ok" << std::endl;
}

void CLWrapper::createZeroCopyObject_WriteFlag_OnDevice() {
    if(onDevice) {
        throw std::runtime_error("createOnDevice(): already on device");
    }
//        std::cout << "creating buffer on device of " << N << " elements" << std::endl;
    devicearray = clCreateBuffer(*(cl->context), CL_MEM_WRITE_ONLY |CL_MEM_ALLOC_HOST_PTR, getElementSize() * N, 0, &error);
    cl->checkError(error);
    onDevice = true;
    deviceDirty = false;
//        std::cout << "... created ok" << std::endl;
}

void CLWrapper::createZeroCopyObject_HostNotAccessFlag_OnDevice() {
    if(onDevice) {
        throw std::runtime_error("createOnDevice(): already on device");
    }
//        std::cout << "creating buffer on device of " << N << " elements" << std::endl;
    devicearray = clCreateBuffer(*(cl->context), CL_MEM_HOST_NO_ACCESS |CL_MEM_ALLOC_HOST_PTR, getElementSize() * N, 0, &error);
    cl->checkError(error);
    onDevice = true;
    deviceDirty = false;
//        std::cout << "... created ok" << std::endl;
}

void CLWrapper::createZeroCopyObject_ReadFlag_OnDevice() {
    if(onDevice) {
        throw std::runtime_error("createOnDevice(): already on device");
    }
//        std::cout << "creating buffer on device of " << N << " elements" << std::endl;
    devicearray = clCreateBuffer(*(cl->context), CL_MEM_READ_ONLY |CL_MEM_ALLOC_HOST_PTR, getElementSize() * N, 0, &error);
    cl->checkError(error);
    onDevice = true;
    deviceDirty = false;
//        std::cout << "... created ok" << std::endl;
}
void CLWrapper::createZeroCopyObject_ReadWriteFlag_OnDevice() {
    if(onDevice) {
        throw std::runtime_error("createOnDevice(): already on device");
    }
//        std::cout << "creating buffer on device of " << N << " elements" << std::endl;
    devicearray = clCreateBuffer(*(cl->context), CL_MEM_READ_WRITE |CL_MEM_ALLOC_HOST_PTR, getElementSize() * N, 0, &error);
    cl->checkError(error);
    onDevice = true;
    deviceDirty = false;
//        std::cout << "... created ok" << std::endl;
}

void CLWrapper::copyToHost(void * buffer) {
    if(!onDevice) {
        throw std::runtime_error("copyToHost(): not on device");
    }
//    cl->finish();
    cl_event event = NULL;
    error = clEnqueueReadBuffer(*(cl->queue), devicearray, CL_TRUE, 0, getElementSize() * N, buffer, 0, NULL, &event);
    cl->checkError(error);
    cl_int err = clWaitForEvents(1, &event);
    clReleaseEvent(event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("wait for event on copytohost failed with " + easycl::toString(err) );
    }
    deviceDirty = false;
}

void CLWrapper::copyToHost() {
    if(!onDevice) {
        throw std::runtime_error("copyToHost(): not on device");
    }
//    cl->finish();
    cl_event event = NULL;
    error = clEnqueueReadBuffer(*(cl->queue), devicearray, CL_TRUE, 0, getElementSize() * N, getHostArray(), 0, NULL, &event);
    cl->checkError(error);
    cl_int err = clWaitForEvents(1, &event);
    clReleaseEvent(event);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("wait for event on copytohost failed with " + easycl::toString(err) );
    }
    deviceDirty = false;
}
cl_mem CLWrapper::getBuffer() { // be careful!
    return devicearray;
}
void CLWrapper::copyToDevice() {
    if(!onHost) {
    	LOGI("not on host");
        throw std::runtime_error("copyToDevice(): not on host");
    }
    if(!onDevice) {

        createOnDevice();
    }
    error = clEnqueueWriteBuffer(*(cl->queue), devicearray, CL_TRUE, 0, getElementSize() * N, getHostArrayConst(), 0, NULL, NULL);
    cl->checkError(error);
    deviceDirty = false;
}
void CLWrapper::copyToDevice_ZeroCopyObject_WriteFlag(const void * buffer) {
    if(!onHost) {
    	LOGI("not on host: problem");
        throw std::runtime_error("copyToDevice(): not on host");
    }
    if(!onDevice) {
        LOGI("not onDevice: we have to create it");
        createZeroCopyObject_WriteFlag_OnDevice();
    }
	float * mappedMatReflector = (float *)clEnqueueMapBuffer (*(cl->queue),devicearray,CL_TRUE,  CL_MAP_WRITE,0,getElementSize() * N, 0, NULL, NULL, NULL);
	std::memcpy(mappedMatReflector, buffer, getElementSize() * N);
	clEnqueueUnmapMemObject(*(cl->queue),devicearray, mappedMatReflector, 0, NULL, NULL);

    //error = clEnqueueWriteBuffer(*(cl->queue), devicearray, CL_TRUE, 0, getElementSize() * N, buffer, 0, NULL, NULL);
    //cl->checkError(error);
    deviceDirty = false;
}

float * CLWrapper::map_ZeroCopyObject_WriteFlag() {
	// note olivier: do not forget to unmap the buffer

	float * mappedMatReflector = (float *)clEnqueueMapBuffer (*(cl->queue),devicearray,CL_TRUE,  CL_MAP_WRITE,0,getElementSize() * N, 0, NULL, NULL, NULL);
return mappedMatReflector;
}

float * CLWrapper::map_ZeroCopyObject_ReadFlag() {
	// note olivier: do not forget to unmap the buffer
	float * mappedMatReflector = (float *)clEnqueueMapBuffer (*(cl->queue),devicearray,CL_TRUE,  CL_MAP_READ,0,getElementSize() * N, 0, NULL, NULL, NULL);

return mappedMatReflector;
}
void CLWrapper::unMap_ZeroCopyObject_ReadFlag(float *ptr) {
	// note olivier: do not forget to unmap the buffer

	clEnqueueUnmapMemObject(*(cl->queue),devicearray, ptr, 0, NULL, NULL);

}


void CLWrapper::unMap_ZeroCopyObject_WriteFlag(float *ptr) {
	// note olivier: do not forget to unmap the buffer

	clEnqueueUnmapMemObject(*(cl->queue),devicearray, ptr, 0, NULL, NULL);

}

void CLWrapper::copyToDevice(const void * buffer) {
    if(!onHost) {
    	LOGI("not on host: problem");
        throw std::runtime_error("copyToDevice(): not on host");
    }
    if(!onDevice) {
        LOGI("not onDevice: we have to create it");
        createOnDevice();
    }
    error = clEnqueueWriteBuffer(*(cl->queue), devicearray, CL_TRUE, 0, getElementSize() * N, buffer, 0, NULL, NULL);
    cl->checkError(error);
    deviceDirty = false;
}


int CLWrapper::size() {
    return N;
}
bool CLWrapper::isOnHost(){
    return onHost;
}
bool CLWrapper::isOnDevice(){
    return onDevice;
}
bool CLWrapper::isDeviceDirty() {
    return deviceDirty;
}
void CLWrapper::markDeviceDirty() {
    deviceDirty = true;
}
void CLWrapper::copyTo(CLWrapper *target) {
    if(size() != target->size()) {
        throw std::runtime_error("copyTo: array size mismatch between source and target CLWrapper objects " + easycl::toString(size()) + " vs " + easycl::toString(target->size()));
    }
  copyTo(target, 0, 0, N);
}
void CLWrapper::copyTo(CLWrapper *target, int srcOffset, int dstOffset, int count) {
    if(!onDevice) {
        throw std::runtime_error("Must have called copyToDevice() or createOnDevice() before calling copyTo(CLWrapper*)");
    }
    if(!target->onDevice) {
        throw std::runtime_error("Must have called copyToDevice() or createOnDevice() on target before calling copyTo(target)");
    }
    if(srcOffset + count > N) {
      throw std::runtime_error("copyTo: not enough source elements, given offset " + easycl::toString(srcOffset) + " and count " + easycl::toString(count));
    }
    if(dstOffset + count > target->N) {
      throw std::runtime_error("copyTo: not enough destation elements, given offset " + easycl::toString(dstOffset) + " and count " + easycl::toString(count));
    }
    if(getElementSize() != target->getElementSize()) {
        throw std::runtime_error("copyTo: element size mismatch between source and target CLWrapper objects");
    }
    // can assume that we have our data on the device now, because of if check
    // just now
    // we will also assume that destination CLWrapper* is valid
//    cl_event event = NULL;
    cl_int err = clEnqueueCopyBuffer(*(cl->queue), devicearray, target->devicearray, 
        srcOffset * getElementSize(), dstOffset * getElementSize(), count * getElementSize(),
        0, NULL, 0);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("copyTo failed with " + easycl::toString(err) );
    }
    else {
        /* Wait for calculations to be finished. */
//        err = clWaitForEvents(1, &event);
    }
//    clReleaseEvent(event);
    target->markDeviceDirty();
}

