using gfoidl.Stochastics.Native;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Native.GpuTests
{
    [TestFixture, Explicit("GPU CUDA must be available")]
    public class CalculateSampleStats
    {
        [Test]
        public void Values_given___correct_stats()
        {
            Assume.That(Gpu.IsAvailable, "GPU is not available");

            double[] values = { 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999 };

            var sample = new Sample(values);

            Gpu.CalculateSampleStats(sample);

            Assert.Multiple(() =>
            {
                // Expected values calculated with gnuplot 5.0 patchlevel 1
                Assert.AreEqual(51.9500 , sample.Mean                   , 1e-3, nameof(sample.Mean));
                Assert.AreEqual(217.2718, sample.StandardDeviation      , 1e-3, nameof(sample.StandardDeviation));
                Assert.AreEqual(222.9162, sample.SampleStandardDeviation, 1e-3, nameof(sample.SampleStandardDeviation));
                Assert.AreEqual(4.1293  , sample.Skewness               , 1e-3, nameof(sample.Skewness));
                Assert.AreEqual(18.0514 , sample.Kurtosis               , 1e-3, nameof(sample.Kurtosis));
                Assert.AreEqual(94.7050 , sample.Delta                  , 1e-3, nameof(sample.Delta));
            });
        }
    }
}
