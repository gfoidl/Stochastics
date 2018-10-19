//#define SMALL_SAMPLE
//#define MEDIUM_SAMPLE
//-----------------------------------------------------------------------------
#include "gpu_core.h"
#include <cassert>
#include <iostream>
//-----------------------------------------------------------------------------
using std::cout;
using std::cerr;
using std::endl;
//-----------------------------------------------------------------------------
int main()
{
#if defined(SMALL_SAMPLE)
    const int N          = 3;
    double sample[3]     = {1, 2, 3};
    double avgExpected   = 2;
    double deltaExpected = 2.0 / 3;
#elif defined(MEDIUM_SAMPLE)
    const int N          = 20;
    double sample[]      = {0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999};
    double avgExpected   = 51.95;
    double deltaExpected = 94.705;
#else
    const int N          = 2000000;
    double* sample       = new double[N];
    double avgExpected   = 0;
    double deltaExpected = 0;
    
    for (int i = 0; i < N; ++i)
    {
        double t  = (double)rand() / RAND_MAX;
        sample[i] = t;
        avgExpected += t;
    }

    avgExpected /= N;

    for (int i = 0; i < N; ++i)
        deltaExpected += abs(sample[i] - avgExpected);

    deltaExpected /= N;
#endif

    SampleStats sampleStats;
    int errorCode = gpu_sample_calc_stats(sample, N, &sampleStats);

    cout << endl;
    cout << "CUDA errorcode: " << errorCode << endl;

    if (errorCode != 0)
    {
        const char* msg = gpu_get_error_string(errorCode);
        cerr << msg << endl;
        return 1;
    }

    cout << endl;
    cout << "avg" << endl;
    cout << "expected: " << avgExpected << endl;
    cout << "actual:   " << sampleStats.Mean << endl;
    assert(abs(avgExpected - sampleStats.Mean) < 1e-3);

    cout << endl;
    cout << "delta" << endl;
    cout << "expected: " << deltaExpected << endl;
    cout << "actual:   " << sampleStats.Delta << endl;
    assert(abs(deltaExpected - sampleStats.Delta) < 1e-3);
}
