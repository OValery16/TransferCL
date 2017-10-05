// Copyright Olivier Valery 2017 olivier.valery92 at gmail.com
// this work is based on DeepCL: a project of Hugh Perkins hughperkins at gmail
//
// This Source Code Form is subject to the terms of the Mozilla Public License,
// v. 2.0. If a copy of the MPL was not distributed with this file, You can
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#if defined(_WIN32) 
# if defined(TransferCL_EXPORTS)
#  define TransferCL_EXPORT __declspec(dllexport)
# else
#  define TransferCL_EXPORT __declspec(dllimport)
# endif // TransferCL_EXPORTS
#else // _WIN32
# define TransferCL_EXPORT
#endif


#define PUBLICAPI

#define PUBLIC
#define PROTECTED
#define PRIVATE

typedef unsigned char uchar;

typedef long long int64;
typedef int int32;

#include "dependencies.h"

