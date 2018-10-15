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
    const int N           = 3;
    double sample[3]      = {1, 2, 3};
    const double expected = 2;
#elif defined(MEDIUM_SAMPLE)
    const int N           = 20;
    double sample[]       = {0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999};
    const double expected = 51.95;
#else
    const int N     = 2000000;
    double* sample  = new double[N];
    double expected = 0;
    
    for (int i = 0; i < N; ++i)
    {
        double t  = (double)rand() / RAND_MAX;
        sample[i] = t;
        expected += t;
    }

    expected /= N;
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

    cout << "expected: " << expected << endl;
    cout << "avg:      " << sampleStats.Mean << endl;
    //assert(abs(expected - sampleStats.Mean) < 1e-3);
}
