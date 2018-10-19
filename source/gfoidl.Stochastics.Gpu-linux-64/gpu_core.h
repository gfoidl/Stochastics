#pragma once
//-----------------------------------------------------------------------------
#include "dll.h"
#include "SampleStats.h"
//-----------------------------------------------------------------------------
BEGIN_EXTERN_C

GPU_API const bool gpu_available();
GPU_API const char* gpu_get_error_string(const int errorCode);

GPU_API const int gpu_sample_calc_stats(double* sample, const int sampleSize, SampleStats* sampleStats);

END_EXTERN_C
