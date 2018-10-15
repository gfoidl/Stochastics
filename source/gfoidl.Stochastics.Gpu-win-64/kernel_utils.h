#pragma once
//-----------------------------------------------------------------------------
#include <cuda_runtime.h>
//-----------------------------------------------------------------------------
namespace Kernel
{
    namespace Utils
    {
        struct TwoDoubles
        {
            double A;
            double B;

            __device__ TwoDoubles() {}
            __device__ TwoDoubles(const double a, const double b)
            {
                this->A = a;
                this->B = b;
            }
        };
        //---------------------------------------------------------------------
        __device__ double WarpReduceSum(double value);
        __device__ TwoDoubles WarpReduceSum(TwoDoubles twoDoubles);

        __device__ double BlockReduceSum(double value);
        __device__ TwoDoubles BlockReduceSum(TwoDoubles twoDoubles);

        __device__ void ReduceSum(double value, double* result);
        __device__ void ReduceSum(TwoDoubles twoDoubles, TwoDoubles* result);
        __device__ void ReduceSum(const double* in, const int n, double* result);

    }
}
