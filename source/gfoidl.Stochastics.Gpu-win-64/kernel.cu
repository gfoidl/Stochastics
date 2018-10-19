#include "kernel.h"
#include "kernel_utils.h"
//-----------------------------------------------------------------------------
namespace Kernel
{
    __global__
    void CalculateAverageAndVarianceCore(const double* sample, const int n, SampleStats* sampleStats)
    {
        const int index  = blockDim.x * blockIdx.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;

        double avg      = 0;
        double variance = 0;

        for (int i = index; i < n; i += stride)
        {
            avg      += sample[i];
            variance += sample[i] * sample[i];
        }

        Utils::TwoDoubles twoDoubles {avg, variance};
        twoDoubles = Utils::BlockReduceSum(twoDoubles);

        // Final sum in first thread of each block
        if (threadIdx.x == 0)
        {
            atomicAdd(&sampleStats->Mean        , twoDoubles.A);
            atomicAdd(&sampleStats->VarianceCore, twoDoubles.B);
        }
    }
    //-----------------------------------------------------------------------------
    __global__
    void CalculateAverageAndVarianceCoreFinal(SampleStats* sampleStats, const int n)
    {
        const int index = blockDim.x * blockIdx.x + threadIdx.x;

        if (index == 0)
        {
            double avg                 = sampleStats->Mean / n;
            sampleStats->Mean          = avg;
            sampleStats->VarianceCore -= n * avg*avg;
        }
    }
    //-----------------------------------------------------------------------------
    __global__
    void CalculateDeltaSkewnessKurtosis(const double* sample, const int n, SampleStats* sampleStats)
    {
        const int index  = blockDim.x * blockIdx.x + threadIdx.x;
        const int stride = gridDim.x * blockDim.x;

        double avg      = sampleStats->Mean;
        double delta    = 0;
        double skewness = 0;
        double kurtosis = 0;

        for (int i = index; i < n; i += stride)
        {
            double t = sample[i] - avg;
            double t1 = t * t*t;

            delta    += abs(t);
            skewness += t1;
            kurtosis += t1 * t;
        }

        Utils::ThreeDoubles threeDoubles {delta, skewness,kurtosis};
        threeDoubles = Utils::BlockReduceSum(threeDoubles);

        // Final sum in first thread of each block
        if (threadIdx.x == 0)
        {
            atomicAdd(&sampleStats->Delta, threeDoubles.A);
            atomicAdd(&sampleStats->Skewness, threeDoubles.B);
            atomicAdd(&sampleStats->Kurtosis, threeDoubles.C);
        }
    }
}
