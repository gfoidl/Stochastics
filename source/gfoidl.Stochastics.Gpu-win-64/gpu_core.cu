#include "gpu_core.h"
#include <cuda_runtime.h>
#include "kernel.h"

#if defined(DEBUG) || defined(_DEBUG)
    #include <stdio.h>
    #include <assert.h>
#endif
//-----------------------------------------------------------------------------
// Forward declarations
inline cudaError_t checkCuda(cudaError_t result);
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
    double*      deviceSample;
    SampleStats* deviceSampleStats;

    try
    {
        checkCuda(cudaMalloc(&deviceSample, sizeof(double) * sampleSize));
        checkCuda(cudaMalloc(&deviceSampleStats, sizeof(SampleStats)));

        checkCuda(cudaMemcpy(deviceSample, sample, sizeof(double) * sampleSize, cudaMemcpyHostToDevice));
        checkCuda(cudaMemset(deviceSampleStats, 0, sizeof(SampleStats)));

        const int blockSize = 256;
        int numBlocks       = (sampleSize + blockSize - 1) / blockSize;

#if defined(DEBUG) || defined(_DEBUG)
        printf("blockSize: %d\nnumBlocks: %d\n", blockSize, numBlocks);
#endif

        // For final fixup of values a separate kernel is queued to the device.
        // Otherwise there's no way of syncing all threads in the grid.
        Kernel::CalculateAverageAndVarianceCore<<<numBlocks, blockSize>>>(deviceSample, sampleSize, deviceSampleStats);
        Kernel::CalculateAverageAndVarianceCoreFinal<<<1, 1>>>(deviceSampleStats, sampleSize);

        //checkCuda(cudaDeviceSynchronize());       // not necessary
        checkCuda(cudaMemcpy(sampleStats, deviceSampleStats, sizeof(SampleStats), cudaMemcpyDeviceToHost));

        checkCuda(cudaFree(deviceSample));
        checkCuda(cudaFree(deviceSampleStats));
    }
    catch (const int e)
    {
        return e;
    }

    return 0;
}
//-----------------------------------------------------------------------------
cudaError_t checkCuda(cudaError_t result)
{
    if (result != cudaSuccess)
        throw static_cast<int>(result);

    return result;
}
