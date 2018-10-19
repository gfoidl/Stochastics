#include "stdafx.h"
#include "TestHelper.h"
#include "gpu_core.h"
//-----------------------------------------------------------------------------
using namespace Microsoft::VisualStudio::CppUnitTestFramework;
//-----------------------------------------------------------------------------
namespace gfoidlStochasticsGpuwin64Tests
{
    TEST_CLASS(gpu_core_Tests)
    {
    private:
        static double _sample[];

    public:
        TEST_METHOD(gpu_available___true)
        {
            bool actual = gpu_available();

            Assert::IsTrue(actual);
        }
        //---------------------------------------------------------------------
        TEST_METHOD(gpu_sample_calc_mean___OK)
        {
            const int N     = 1000000;
            double* sample  = new double[N];
            double expected = 0.0;

            for (int i = 0; i < N; ++i)
            {
                double t  = (double)rand() / RAND_MAX;
                sample[i] = t;
                expected += t;
            }
            expected /= N;

            SampleStats sampleStats;
            int errorCode = gpu_sample_calc_stats(sample, N, &sampleStats);

            TestHelper::FailIfError(errorCode);

            Assert::AreEqual(expected, sampleStats.Mean, 1e-3);

            delete[] sample;
        }
        //---------------------------------------------------------------------
        TEST_METHOD(gpu_sample_calc_stats___OK)
        {
            SampleStats sampleStats;
            int errorCode = gpu_sample_calc_stats(_sample, 20, &sampleStats);

            TestHelper::FailIfError(errorCode);

            // Expected values calculated with gnuplot 5.0 patchlevel 1
            double standardDeviation       = sqrt(sampleStats.VarianceCore / 20);
            double sampleStandardDeviation = sqrt(sampleStats.VarianceCore / (20 - 1));

            Assert::AreEqual(51.9500 , sampleStats.Mean       , 1e-3);
            Assert::AreEqual(217.2718, standardDeviation      , 1e-3);
            Assert::AreEqual(222.9162, sampleStandardDeviation, 1e-3);
            Assert::AreEqual(94.7050 , sampleStats.Delta      , 1e-3);
            //Assert::AreEqual(4.1293  , sampleStats.Skewness   , 1e-3);
            //Assert::AreEqual(18.0514 , sampleStats.Kurtosis   , 1e-3);
        }
    };
    //-------------------------------------------------------------------------
    double gpu_core_Tests::_sample[] = {0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999};
}
