using System;
using gfoidl.Stochastics.Statistics;
using NUnit.Framework;

namespace gfoidl.Stochastics.Tests.Statistics.SampleTests
{
    [TestFixture(10)]
    [TestFixture(100)]
    [TestFixture(1_000)]
    [TestFixture(10_000)]
    [TestFixture(100_000)]
    [TestFixture(1_000_000)]
    public class SkewnessAndKurtosis
    {
        private readonly int _size;
        //---------------------------------------------------------------------
        public SkewnessAndKurtosis(int size) => _size = size;
        //---------------------------------------------------------------------
        [Test]
        public void Simd_and_ParallelizedSimd_produce_same_result()
        {
            var values = new double[_size];
            var rnd    = new Random();

            for (int i = 0; i < values.Length; ++i)
                values[i] = rnd.NextDouble();

            var sut = new Sample(values);

            (double skewness, double kurtosis) actual1 = sut.CalculateSkewnessAndKurtosisSimd();
            (double skewness, double kurtosis) actual2 = sut.CalculateSkewnessAndKurtosisParallelizedSimd();

            Assert.AreEqual(actual1.skewness, actual2.skewness, 1e-7);
            Assert.AreEqual(actual1.kurtosis, actual2.kurtosis, 1e-7);
        }
    }
}