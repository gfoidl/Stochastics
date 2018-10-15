#include "gpu_core.h"
#include <cuda_runtime.h>
//-----------------------------------------------------------------------------
const bool gpu_available()
{
    int deviceCount;
    cudaError_t errorId = cudaGetDeviceCount(&deviceCount);

    return errorId == cudaSuccess
        && deviceCount > 0;
}
//-----------------------------------------------------------------------------
const char* gpu_get_error_string(const int errorCode)
{
    return cudaGetErrorString(static_cast<cudaError>(errorCode));
}
//-----------------------------------------------------------------------------
const int gpu_sample_calc_stats(double* sample, const int sampleSize, SampleStats* sampleStats)
{
    sampleStats->Delta = 1;
    sampleStats->Kurtosis = 2;
    sampleStats->Max = 3;
    sampleStats->Mean = 4;
    sampleStats->Min = 5;
    sampleStats->Skewness = 6;
    sampleStats->VarianceCore = 7;

    return 0;
}
