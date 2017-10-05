// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com 
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.


#include <cstdio>
#include <stdexcept>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

#ifdef _WIN32
#include "windows.h"
#else
#include <sys/stat.h>
#endif

#include "FileHelper.h"

#undef VIRTUAL
#define VIRTUAL 
#undef STATIC
#define STATIC

PUBLIC STATIC char *FileHelper::readBinary(std::string filepath, long *p_filesize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: readBinary");
#endif


    std::string localPath = localizePath(filepath);
    std::ifstream file(localPath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!file.is_open()) {
    	LOGI("couldnt open file %s",localPath.c_str());
        throw std::runtime_error("couldnt open file " + localPath);
    }
    *p_filesize = static_cast<long>(file.tellg());
//    std::cout << " filesize " << *p_filesize << std::endl;
    char *data = new char[*p_filesize];
    file.seekg(0, std::ios::beg);
    if(!file.read(data, *p_filesize)) {
    	LOGI("failed to read from %s",localPath.c_str());
        throw std::runtime_error("failed to read from " + localPath);
    }
    file.close();
    return data;
}
PUBLIC STATIC long FileHelper::getFilesize(std::string filepath) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: getFilesize");
#endif


    std::ifstream in(localizePath(filepath).c_str(), std::ifstream::ate | std::ifstream::binary);
    return static_cast<long>(in.tellg()); 
}
PUBLIC STATIC char *FileHelper::readBinaryChunk(std::string filepath, long start, long length) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: readBinaryChunk");
#endif

//it crash here because it try to extract 1024 characters but there are only 150

    std::string localPath = localizePath(filepath);

//    LOGI( "localPath %s",localPath.c_str());
    std::ifstream file(localPath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
//    LOGI( "file define");
    if(!file.is_open()) {
    	LOGE( "readBinaryChunk: error ");
        throw std::runtime_error("failed to open file: " + localPath);
    }
//    LOGI( "file opened");
    std::streampos size = file.tellg();
//    LOGI( "size %d",(int)size);
    char *data = new char [size];
    file.seekg (0, std::ios::beg);
    file.read (data, size);
    file.close();

//    char *data = new char[contentfile.length()+1];
//    strcpy(data,contentfile.c_str());
//    LOGE("contentfile=%s", contentfile.c_str());
    //data=contentfile.c_str();

//    file.seekg(start, std::ios::beg);
//    char *data = new char[length];
//    if(!file.read(data, length)) {
//    	LOGE( "readBinaryChunk: error, failed to read");
//        throw std::runtime_error("failed to read from " + localPath);
//    }
    //LOGE("List of files in the set (defined in the manifest):");
    //LOGE("data=%s", data);
    file.close();
    return data;
}
// need to allocate targetArray yourself, beforehand
PUBLIC STATIC void FileHelper::readBinaryChunk(char *targetArray, std::string filepath, long start, long length) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: readBinaryChunk");
#endif


    std::string localPath = localizePath(filepath);
    std::ifstream file(localPath.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    if(!file.is_open()) {
        throw std::runtime_error("failed to open file: " + localPath);
    }
    file.seekg(start, std::ios::beg);
//        char *data = new char[length];
    if(!file.read(targetArray, length)) {
        throw std::runtime_error("failed to read from " + localPath);
    }
    file.close();
//        return data;
}
PUBLIC STATIC void FileHelper::writeBinary(std::string filepath, char const*data, long filesize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: writeBinary");
#endif


    std::string localPath = localizePath(filepath);
    std::ofstream file(localPath.c_str(), std::ios::out | std::ios::binary);
    if(!file.is_open()) {
         throw std::runtime_error("cannot open file " + localPath);
    }
    if(!file.write((char *)data, filesize) ) {
        throw std::runtime_error("failed to write to " + localPath);
    }
    file.close();
}
PUBLIC STATIC void FileHelper::writeBinaryChunk(std::string filepath, char const*data, long startPos, long filesize) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: writeBinaryChunk");
#endif


    std::string localPath = localizePath(filepath);
    std::ofstream file(localPath.c_str(), std::ios::out | std::ios::binary);
    file.seekp(startPos, std::ios::beg);
    if(!file.is_open()) {
         throw std::runtime_error("cannot open file " + localPath);
    }
    if(!file.write((char *)data, filesize) ) {
        throw std::runtime_error("failed to write to " + localPath);
    }
    file.close();
}
PUBLIC STATIC bool FileHelper::exists(const std::string filepath) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: exists");
#endif


   std::string localPath = localizePath(filepath);
   std::ifstream testifstream(localPath.c_str());
   bool exists = testifstream.good();
   testifstream.close();
   return exists;
}

PUBLIC STATIC void FileHelper::rename(std::string oldname, std::string newname) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: rename");
#endif


    ::rename(localizePath(oldname).c_str(), localizePath(newname).c_str());
}

PUBLIC STATIC void FileHelper::remove(std::string filename) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: remove");
#endif


    ::remove(localizePath(filename).c_str());
}
PUBLIC STATIC std::string FileHelper::localizePath(std::string path) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: string FileHelper::localizePath");
#endif


    std::replace(path.begin(), path.end(), '/', pathSeparator().c_str()[0]);
    //std::cout << "localized path: " << path << std::endl;
    return path;
}
PUBLIC STATIC std::string FileHelper::pathSeparator() {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: string FileHelper::pathSeparator");
#endif


#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
PUBLIC STATIC void FileHelper::createDirectory(std::string path) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: createDirectory");
#endif


    #ifdef _WIN32
        if(CreateDirectory(path.c_str(), NULL) == 0) {
            throw std::runtime_error("Failed to create directory " + path);
        }
    #else
        if(::mkdir(path.c_str(), 0775) == -1  ) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: mkdir(path.c_str(), 0775) == -1  ) {");
#endif


            throw std::runtime_error("Failed to create directory " + path);
        }
    #endif
}
PUBLIC STATIC bool FileHelper::folderExists(std::string path) {
#if TRANSFERCL_VERBOSE == 1
LOGI( "DeepCL/src/util/FileHelper.cpp: folderExists");
#endif


    #ifdef _WIN32
        return GetFileAttributes(path.c_str()) == INVALID_FILE_ATTRIBUTES;
    #else
        struct stat status;
        stat(path.c_str(), &status);
        return S_ISDIR(status.st_mode);
    #endif
}

