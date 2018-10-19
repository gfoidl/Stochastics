#include "kernel_utils.h"
#include <device_launch_parameters.h>

#ifdef _DEBUG
    #include <cassert>
#endif
//-----------------------------------------------------------------------------
// Code for reduction taken from https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
//-----------------------------------------------------------------------------
namespace Kernel
{
    namespace Utils
    {
        using uint = unsigned int;
        const uint FULL_MASK = 0xffffffff;
        //---------------------------------------------------------------------
#if __CUDA_ARCH__ < 600 && false
        __device__
        double atomicAdd(double* address, double val)
        {
            using ulli = unsigned long long int;

            ulli* tmp = reinterpret_cast<ulli*>(address);
            ulli old = *tmp;
            ulli assumed;

            do
            {
                assumed = old;

                old = atomicCAS(tmp, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
            } while (assumed != old);

            return __longlong_as_double(old);
        }
#endif
        //---------------------------------------------------------------------
        __device__
        double WarpReduceSum(double value)
        {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
                value += __shfl_down_sync(FULL_MASK, value, offset);

            return value;
        }
        //---------------------------------------------------------------------
        __device__
        TwoDoubles WarpReduceSum(TwoDoubles twoDoubles)
        {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                twoDoubles.A += __shfl_down_sync(FULL_MASK, twoDoubles.A, offset);
                twoDoubles.B += __shfl_down_sync(FULL_MASK, twoDoubles.B, offset);
            }

            return twoDoubles;
        }
        //---------------------------------------------------------------------
        __device__
        ThreeDoubles WarpReduceSum(ThreeDoubles threeDoubles)
        {
            for (int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                threeDoubles.A += __shfl_down_sync(FULL_MASK, threeDoubles.A, offset);
                threeDoubles.B += __shfl_down_sync(FULL_MASK, threeDoubles.B, offset);
                threeDoubles.C += __shfl_down_sync(FULL_MASK, threeDoubles.C, offset);
            }

            return threeDoubles;
        }
        //---------------------------------------------------------------------
        __device__
        double BlockReduceSum(double value)
        {
#ifdef _DEBUG
            assert(warpSize == 32);
#endif
            static __shared__ double shared[32];
            const int lane   = threadIdx.x & (warpSize - 1);        // threadIdx.x % warpSize
            const int warpId = threadIdx.x / warpSize;

            value = WarpReduceSum(value);

            if (lane == 0)
                shared[warpId] = value;

            __syncthreads();

            // Read from shared memory only if that warp existed
            bool warpExisted = (threadIdx.x < blockDim.x / warpSize) || (threadIdx.x == 0 && blockDim.x == 1);
            value = warpExisted ? shared[lane] : 0;

            // Final reduce within first warp
            if (warpId == 0)
                value = WarpReduceSum(value);

            return value;
        }
        //---------------------------------------------------------------------
        __device__
        TwoDoubles BlockReduceSum(TwoDoubles twoDoubles)
        {
#ifdef _DEBUG
            assert(warpSize == 32);
#endif
            static __shared__ TwoDoubles shared[32];
            const int lane   = threadIdx.x & (warpSize - 1);        // threadIdx.x % warpSize
            const int warpId = threadIdx.x / warpSize;

            twoDoubles = WarpReduceSum(twoDoubles);

            if (lane == 0)
                shared[warpId] = twoDoubles;

            __syncthreads();

            // Read from shared memory only if that warp existed
            bool warpExisted = (threadIdx.x < blockDim.x / warpSize) || (threadIdx.x == 0 && blockDim.x == 1);
            twoDoubles = warpExisted ? shared[lane] : TwoDoubles {0,0};

            // Final reduce within first warp
            if (warpId == 0)
                twoDoubles = WarpReduceSum(twoDoubles);

            return twoDoubles;
        }
        //---------------------------------------------------------------------
        __device__
        ThreeDoubles BlockReduceSum(ThreeDoubles threeDoubles)
        {
#ifdef _DEBUG
            assert(warpSize == 32);
#endif
            static __shared__ ThreeDoubles shared[32];
            const int lane = threadIdx.x & (warpSize - 1);      // threadIdx.x % warpSize
            const int warpId = threadIdx.x / warpSize;

            threeDoubles = WarpReduceSum(threeDoubles);

            if (lane == 0)
                shared[warpId] = threeDoubles;

            __syncthreads();

            // Read from shared memory only if that warp existed
            bool warpExisted = (threadIdx.x < blockDim.x / warpSize) || (threadIdx.x == 0 && blockDim.x == 1);
            threeDoubles = warpExisted ? shared[lane] : ThreeDoubles {0,0,0};

            // Final reduce within first warp
            if (warpId == 0)
                threeDoubles = WarpReduceSum(threeDoubles);

            return threeDoubles;
        }
        //---------------------------------------------------------------------
        __device__
        void ReduceSum(double value, double* result)
        {
            value = BlockReduceSum(value);

            // Final sum in first thread of each block
            if (threadIdx.x == 0)
                atomicAdd(result, value);
        }
        //---------------------------------------------------------------------
        __device__
        void ReduceSum(TwoDoubles twoDoubles, TwoDoubles* result)
        {
            twoDoubles = BlockReduceSum(twoDoubles);

            // Final sum in first thread of each block
            if (threadIdx.x == 0)
            {
                atomicAdd(&result->A, twoDoubles.A);
                atomicAdd(&result->B, twoDoubles.B);
            }
        }
        //---------------------------------------------------------------------
        __device__
        void ReduceSum(const double* in, const int n, double* result)
        {
            double blockSum = 0;

            const int index  = blockDim.x * blockIdx.x + threadIdx.x;
            const int stride = gridDim.x * blockDim.x;

            for (int i = index; i < n; i += stride)
                blockSum += in[i];

            ReduceSum(blockSum, result);
        }
    }
}
