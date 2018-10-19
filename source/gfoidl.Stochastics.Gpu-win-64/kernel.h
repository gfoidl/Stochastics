#pragma once
//-----------------------------------------------------------------------------
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "SampleStats.h"
//-----------------------------------------------------------------------------
namespace Kernel
{
    __global__ void CalculateAverageAndVarianceCore(const double* sample, const int n, SampleStats* sampleStats);
    __global__ void CalculateAverageAndVarianceCoreFinal(SampleStats* sampleStats, const int n);

    __global__ void CalculateDelta(const double* sample, const int n, SampleStats* sampleStats);
}
